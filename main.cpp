#define CL_TARGET_OPENCL_VERSION 220
#define __CL_ENABLE_EXCEPTIONS

#include <cmath>
#include <numbers>
#include <CL/cl.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <vector>
#include <chrono>

#include "KernelUtils.hpp"

cv::Mat LoadInputImage(const std::string &image_path)
{
    const cv::Mat inputImage = cv::imread(image_path, cv::IMREAD_COLOR);
    if (inputImage.empty())
    {
        throw std::runtime_error("Failed to load input.png");
    }
    return inputImage;
}

void ShowGrayscaleImage(const cv::Mat &img, const std::string &window_name)
{
    cv::Mat outputBGR;
    cv::Mat output_normalized;
    cv::Mat output_color;

    const std::string win_name1 = window_name + "_normalized";
    const std::string win_name2 = window_name + "_colormap";

    cv::normalize(img, output_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::namedWindow(win_name1, cv::WINDOW_NORMAL);
    cv::imshow(win_name1, output_normalized);

    cv::applyColorMap(output_normalized, output_color, cv::COLORMAP_JET);
    cv::namedWindow(win_name2, cv::WINDOW_NORMAL);
    cv::imshow(win_name2, output_color);
}

int main(int argc, char **argv)
{
    toml::table tbl;
    try
    {
        tbl = toml::parse_file("config.toml");
    } catch (const toml::parse_error &err)
    {
        std::cout << "Error parsing file '" << *err.source().path << "':\n"
                << err.description() << "\n (" << err.source().begin << ")\n";
        return 1;
    }

    /// Get parameters
    const auto use_gpu = ConfigGetValue<bool>(tbl, "OpenCL.use_gpu");
    const auto platform_id = ConfigGetValue<size_t>(tbl, "OpenCL.platform_id");
    const auto device_id = ConfigGetValue<size_t>(tbl, "OpenCL.device_id");

    const auto image_path =
            ConfigGetValue<std::string>(tbl, "Hough_transform.image");
    const auto hough_min_radius =
            ConfigGetValue<int>(tbl, "Hough_transform.min_radius");
    const auto hough_max_radius =
            ConfigGetValue<int>(tbl, "Hough_transform.max_radius");
    const auto radius_threshold =
            ConfigGetValue<int>(tbl, "Hough_transform.radius_threshold");
    const auto radius_step =
            ConfigGetValue<int>(tbl, "Hough_transform.radius_step");
    const auto visualize_process =
            ConfigGetValue<int>(tbl, "Hough_transform.visualize_process");
    const auto hough_space_threshold =
            ConfigGetValue<float>(tbl, "Hough_transform.hough_space_threshold");
    const auto canny_threshold1 =
            ConfigGetValue<float>(tbl, "Hough_transform.canny_threshold1");
    const auto canny_threshold2 =
            ConfigGetValue<float>(tbl, "Hough_transform.canny_threshold2");

    const auto device = GetDevice(platform_id, device_id, use_gpu);

    const cl::Context context(device);
    const cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    const std::string kernelSource = ReadKernelFile("cl/FindCircle.cl");

    const cl::Program program(context, kernelSource);
    try
    {
        program.build({device});
    } catch (cl::Error &buildErr)
    {
        std::cout << "Error building: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        throw;
    }
    cv::ocl::setUseOpenCL(true);
    std::cout << "OpenCV OpenCL is enabled." << std::endl;
    cv::ocl::Device dev = cv::ocl::Device::getDefault();
    std::cout << "OpenCV OpenCL device: " << dev.name() << std::endl;

    // Load input image
    cv::Mat input_img = LoadInputImage(image_path);
    cv::resize(input_img, input_img, cv::Size(), 0.1, 0.1);
    // Convert to grayscale
    cv::Mat grayscale_img;
    cv::cvtColor(input_img, grayscale_img, cv::COLOR_BGR2GRAY);
    grayscale_img.convertTo(grayscale_img, CV_8UC1);

    // Extract edge
    cv::Mat input_canny;
    cv::Mat blurred;
    cv::GaussianBlur(grayscale_img, blurred, cv::Size(5,5), 1.4);
    cv::Canny(blurred, input_canny, canny_threshold1, canny_threshold2);

    if (visualize_process)
    {
        cv::imshow("Canny", input_canny);
        auto sum = cv::sum(input_canny);
        std::cout << sum << std::endl;
    }

    cv::Mat circle_accumulator(input_img.rows, input_img.cols, CV_8UC1);

    // Create input/output of opencl images
    cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);

    cl::Image2D inputImageCL(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             format, input_canny.cols, input_canny.rows, 0,
                             input_canny.data);

    // Create output image
    cl::Image2D output_find_circle_cl(context, CL_MEM_READ_WRITE, format, input_canny.cols, input_canny.rows);
    cv::Mat cv_output_find_circle(input_canny.rows, input_canny.cols, CV_8UC1);

    // Create container for centroids
    cl::Image2D output_find_radius_cl(context, CL_MEM_WRITE_ONLY, format, input_canny.cols, input_canny.rows);
    cv::Mat cv_output_find_radius(input_canny.rows, input_canny.cols, CV_8UC1);

    /// Kernel -> function name in program (.cl file)
    cl::Kernel kernel_find_circle(program, "FindCircle2");
    cl::Kernel kernel_find_radius(program, "FindRadius");

    // Enqueue kernel execution
    cl::NDRange globalSize(input_canny.cols, input_canny.rows);

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region{};
    region[0] = static_cast<size_t>(input_canny.cols);
    region[1] = static_cast<size_t>(input_canny.rows);
    region[2] = 1;


    cl::Event event;

    const size_t buffer_bytes = input_canny.cols * input_canny.rows * sizeof(cl_uint);
    cl::Buffer cl_accumulator_buffer(context, CL_MEM_READ_WRITE, buffer_bytes);


    for (auto radius = hough_max_radius; radius >= hough_min_radius; radius -= radius_step)
    {
        const auto start = std::chrono::high_resolution_clock::now();

        kernel_find_circle.setArg(0, inputImageCL);
        kernel_find_circle.setArg(1, radius);
        kernel_find_circle.setArg(2, radius_threshold);
        kernel_find_circle.setArg(3, cl_accumulator_buffer);

        // Find circle Kernel
        queue.enqueueFillBuffer(cl_accumulator_buffer, 0, 0, buffer_bytes);
        queue.enqueueNDRangeKernel(kernel_find_circle, cl::NullRange, globalSize, cl::NullRange, 0, &event);
        (void)event.wait();

        cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double duration_ms = (end_time - start_time) * 1e-6;
        std::cout << "Find circle Kernel execution time: " << duration_ms << " ms" << std::endl;

        // Show result image
        if (visualize_process)
        {
            std::vector<uint8_t> hostBuffer(input_canny.rows * input_canny.cols); // uchar = unsigned char = 8-bit

            queue.enqueueReadBuffer(cl_accumulator_buffer, CL_TRUE, 0, hostBuffer.size(), hostBuffer.data());
            cv::Mat tmp(input_canny.rows, input_canny.cols, CV_8UC1, hostBuffer.data());
            cv::Mat found_circles = tmp.clone();
            ShowGrayscaleImage(found_circles, "FindCircle");
        }

        const uint thresholdValue = radius * std::numbers::pi * 2 * hough_space_threshold;

        // Set kernel arguments
        kernel_find_radius.setArg(0, cl_accumulator_buffer);
        kernel_find_radius.setArg(1, thresholdValue);
        kernel_find_radius.setArg(2, radius + radius_threshold);
        kernel_find_radius.setArg(3, output_find_radius_cl);

        // // Find radius kernel
        queue.enqueueNDRangeKernel(kernel_find_radius, cl::NullRange, globalSize, cl::NullRange, 0, &event);
        event.wait();
        start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        duration_ms = (end_time - start_time) * 1e-6;
        std::cout << "Find radius kernel execution time: " << duration_ms << " ms" << std::endl;

        queue.enqueueReadImage(output_find_radius_cl, CL_TRUE, origin, region, 0, 0, cv_output_find_radius.data);

        if (visualize_process)
            cv::imshow("Centroids", cv_output_find_radius);

        for (int y = 0; y < cv_output_find_radius.rows; ++y)
        {
            for (int x = 0; x < cv_output_find_radius.cols; ++x)
            {
                if (cv_output_find_radius.at<uchar>(y, x) > 0)
                {
                    std::cout << "Found circle at: x=" << x << ", y=" << y << std::endl;
                    int r = 255;
                    int g = 0;
                    int b = 0;
                    cv::circle(input_img, cv::Point(x, y), radius, cv::Scalar(r, g, b), 1);
                }
            }
        }

        if (visualize_process)
        {
            // cv::Mat resize_output;
            // cv::resize(input_img, resize_output, cv::Size(1920, 1080), cv::INTER_LINEAR);
            cv::namedWindow("Detected Circles", cv::WINDOW_NORMAL);
            cv::imshow("Detected Circles", input_img);
            cv::waitKey(1);
        }

        const auto end = std::chrono::high_resolution_clock::now();
        std::cout << "One iteration time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }
    std::cout << "End of run" << std::endl;

    if (!visualize_process)
    {
        cv::imshow("Detected Circles", input_img);
    }

    cv::imwrite("HoughTransformOutputImage.png", input_img);
    cv::waitKey(0);

    return 0;
}
