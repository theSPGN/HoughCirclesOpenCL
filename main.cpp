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
#include <numeric>
#include <queue>

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



void ShowAccumulator(std::size_t n_radius, std::size_t rows, std::size_t cols, const cl::CommandQueue &queue, const cl::Buffer *accumulator)
{
    for (int j = 0; j < n_radius; j++)
    {
        const size_t mem_offset = j * rows * cols;

        std::vector<cl_uint> output_buffer(rows * cols);

        queue.enqueueReadBuffer(*accumulator, CL_TRUE, mem_offset * sizeof(cl_uint), output_buffer.size() * sizeof(cl_uint), output_buffer.data());

        std::vector<uint16_t> change_type(rows * cols, 0);
        std::transform(output_buffer.begin(), output_buffer.end(), change_type.begin(), [](cl_uint value){ return static_cast<int16_t>(value); });

        cv::Mat tmp(static_cast<int>(rows), static_cast<int>(cols), CV_16UC1, change_type.data());

        ShowGrayscaleImage(tmp, "FindCircle");
    }
}


void DrawCircle(std::size_t n_radius, std::size_t rows, std::size_t cols, uint min_radius, int radius_step,
    const cl::CommandQueue &queue, const cl::Buffer *buffer, cv::Mat &input_img, bool show_info)
{
    static constexpr int r = 255;
    static constexpr int g = 0;
    static constexpr int b = 0;

    for (auto idx = 0; idx < n_radius; ++idx)
    {
        const int radius = min_radius + idx * radius_step;

        const int mem_offset = idx * rows * cols;
        std::vector<cl_uint> output_buffer(rows * cols, 0);

        queue.enqueueReadBuffer(*buffer, CL_TRUE, mem_offset * sizeof(cl_uint), output_buffer.size() * sizeof(cl_uint), output_buffer.data());

        std::vector<uint8_t> change_type(output_buffer.size(), 0);
        std::transform(output_buffer.begin(), output_buffer.end(), change_type.begin(), [](cl_uint value){ return static_cast<uint8_t>(value); });

        cv::Mat frame(static_cast<int>(rows), static_cast<int>(cols), CV_8UC1, change_type.data());

        if (show_info)
        {
            const auto non_zero = cv::countNonZero(frame);
            const auto zeros = (rows * cols) - non_zero;

            std::cout << "Radius=" << radius << " zeros=" << zeros << " nonzero=" << non_zero << std::endl;
        }

        for (auto y = 0; y < rows; ++y)
        {
            for (auto x = 0; x < cols; ++x)
            {
                if (input_img.at<int8_t>(y, x) != 0)
                {
                    // std::cout << "Drawin circle on: " << x << "," << y << std::endl;
                    cv::circle(input_img, cv::Point(x, y), radius, cv::Scalar(r, g, b), 1);
                }
            }
        }
        std::cout << "Draw circle end radius: " << radius << std::endl;
    }
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
    const auto profile = ConfigGetValue<bool>(tbl, "OpenCL.profile");

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
    const uint thresholdValue = static_cast<uint>(std::numbers::pi * 2 * hough_space_threshold);

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

    // cv::ocl::setUseOpenCL(true);
    // std::cout << "OpenCV OpenCL is enabled." << std::endl;
    // cv::ocl::Device dev = cv::ocl::Device::getDefault();
    // std::cout << "OpenCV OpenCL device: " << dev.name() << std::endl;

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
    }

    cv::Mat circle_accumulator(input_img.rows, input_img.cols, CV_8UC1);

    // Create input/output of opencl images
    cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);

    CV_Assert(input_canny.isContinuous());
    cl::Image2D inputImageCL(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             format, input_canny.cols, input_canny.rows, 0,
                             input_canny.data);

    // Create output image
    cl::Image2D output_find_circle_cl(context, CL_MEM_READ_WRITE, format, input_canny.cols, input_canny.rows);

    // Create container for centroids
    cl::Image2D output_find_radius_cl(context, CL_MEM_WRITE_ONLY, format, input_canny.cols, input_canny.rows);

    /// Kernel -> function name in program (.cl file)
    cl::Kernel kernel_find_circle(program, "FindCircle3");
    cl::Kernel kernel_find_radius(program, "FindRadius2");

    // Query maximum memory allocation size
    const cl_ulong max_alloc_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    std::cout << "Device max alloc size: " << max_alloc_size << std::endl;

    const std::size_t img_size = input_canny.cols * input_canny.rows;
    const std::size_t n_radius = ceil(static_cast<float>(hough_max_radius - hough_min_radius + radius_step - 1) / static_cast<float>(radius_step));
    // const std::size_t size_3d =  (max_alloc_size) / (input_canny.cols * input_canny.rows * 2 * 2); // Radius -> 3d gpu size
    const std::size_t size_3d = 16; // Radius -> 3d gpu size
    std::cout << "3D dimension size: " << size_3d << std::endl;

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region{};
    region[0] = static_cast<size_t>(input_canny.cols);
    region[1] = static_cast<size_t>(input_canny.rows);
    region[2] = 1;

    if (visualize_process)
        cv::imshow("InputImage", input_img);

    cl::Event event2;
    std::vector<cl::Event> depends_event1{1};

    const std::size_t buffer_bytes = static_cast<std::size_t>(size_3d) *
        static_cast<std::size_t>(input_canny.cols) * static_cast<std::size_t>(input_canny.rows) * sizeof(cl_uint);

    cl::Buffer cl_accumulator_buffer1(context, CL_MEM_READ_WRITE, buffer_bytes);
    cl::Buffer cl_accumulator_buffer2(context, CL_MEM_READ_WRITE, buffer_bytes);

    cl::Buffer cl_output_buffer1(context, CL_MEM_READ_WRITE, buffer_bytes);
    cl::Buffer cl_output_buffer2(context, CL_MEM_READ_WRITE, buffer_bytes);

    cl::Buffer *curr_buffer_accum = &cl_accumulator_buffer1;
    cl::Buffer *prev_buffer_accum = &cl_accumulator_buffer2;

    cl::Buffer *curr_buffer_output = &cl_output_buffer1;
    cl::Buffer *prev_buffer_output = &cl_output_buffer2;

    auto OpenCLProfile = [](const cl::Event &event, std::string_view kernel_name){
        (void)event.wait();
        const cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        const cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        const double duration_ms = (end_time - start_time) * 1e-6;
        std::cout << kernel_name << ", execution time: " << duration_ms << " ms" << std::endl;
    };


    auto LaunchKernels = [&](std::size_t opencl_3d, uint curr_min_radius) {
        cl::NDRange globalSize(input_canny.cols, input_canny.rows, opencl_3d);

        // Find circle kernel
        kernel_find_circle.setArg(0, inputImageCL);
        kernel_find_circle.setArg(1, curr_min_radius);
        kernel_find_circle.setArg(2, radius_step);
        kernel_find_circle.setArg(3, radius_threshold);
        kernel_find_circle.setArg(4, *curr_buffer_accum);

        queue.enqueueNDRangeKernel(kernel_find_circle, cl::NullRange, globalSize, cl::NullRange, 0, depends_event1.data());
        if (profile)
            OpenCLProfile(depends_event1[0], "FindCircle");

        // Find radius kernel
        kernel_find_radius.setArg(0, *curr_buffer_accum);
        kernel_find_radius.setArg(1, thresholdValue);
        kernel_find_radius.setArg(2, curr_min_radius);
        kernel_find_radius.setArg(3, radius_threshold);
        kernel_find_radius.setArg(4, *curr_buffer_output);

        queue.enqueueNDRangeKernel(kernel_find_radius, cl::NullRange, globalSize, cl::NullRange, &depends_event1, &event2);
        if (profile)
            OpenCLProfile(event2, "FindRadius");
    };

    /// Program execution
    const auto total_start = std::chrono::high_resolution_clock::now();
    {
        queue.enqueueFillBuffer(*curr_buffer_accum, 0, 0, buffer_bytes);
        queue.enqueueFillBuffer(*prev_buffer_accum, 0, 0, buffer_bytes);
        queue.enqueueFillBuffer(*curr_buffer_output, 0, 0, buffer_bytes);
        queue.enqueueFillBuffer(*prev_buffer_output, 0, 0, buffer_bytes);
        cl::finish();

        std::size_t i = 0;
        std::size_t opencl_3d = (i + size_3d) <= n_radius ? size_3d: (n_radius - i);

        auto start = std::chrono::high_resolution_clock::now();
        uint curr_min_radius = hough_min_radius + i * size_3d;

        LaunchKernels(opencl_3d, curr_min_radius);

        (void)event2.wait();
        i += size_3d;

        std::size_t prev_opencl_3d = opencl_3d;
        uint prev_min_radius = hough_min_radius + i * size_3d;

        for (; i < n_radius; i += size_3d)
        {
            std::swap(curr_buffer_accum, prev_buffer_accum);
            std::swap(curr_buffer_output, prev_buffer_output);

            opencl_3d = (i + size_3d) <= n_radius ? size_3d: (n_radius - i);
            curr_min_radius = hough_min_radius + i * size_3d;

            LaunchKernels(opencl_3d, curr_min_radius);

            std::cout << "before accumualtor" << std::endl;
            if (visualize_process)
            {
                ShowAccumulator(prev_opencl_3d, input_canny.rows, input_canny.cols, queue, prev_buffer_accum);
            }

            std::cout << "Draw circle" << std::endl;
            DrawCircle(prev_opencl_3d, input_canny.rows, input_canny.cols, prev_min_radius, radius_step,
                queue, prev_buffer_output, input_img, visualize_process);

            const auto end = std::chrono::high_resolution_clock::now();
            std::cout << "One iteration time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;


            // Wait for next iteration of openCL
            cl_int status = event2.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
            if (status != CL_COMPLETE) {
                event2.wait();
            }

            start = std::chrono::high_resolution_clock::now();

            prev_opencl_3d = opencl_3d;
            prev_min_radius = curr_min_radius;
        }

        // Last iteration of circle only circle
        if (visualize_process)
        {
            ShowAccumulator(prev_opencl_3d, input_canny.rows, input_canny.cols, queue, curr_buffer_accum);
        }

        DrawCircle(prev_opencl_3d, input_canny.rows, input_canny.cols, prev_min_radius, radius_step,
            queue, curr_buffer_output, input_img, visualize_process);

        const auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Last iteration time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;
    }

    std::cout << "==========End of run==========" << std::endl;
    const auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count() << " s." << std::endl;

    cv::namedWindow("Detected Circles", cv::WINDOW_NORMAL);
    cv::imshow("Detected Circles", input_img);
    cv::waitKey(1);

    cv::imwrite("HoughTransformOutputImage.png", input_img);
    cv::waitKey(0);

    return 0;
}
