#define CL_TARGET_OPENCL_VERSION 220
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "KernelUtils.hpp"


cv::Mat LoadInputImage(const std::string &image_path)
{
    const cv::Mat inputImage = cv::imread(image_path, cv::IMREAD_COLOR_BGR);
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
    }
    catch (const toml::parse_error &err)
    {
        std::cout << "Error parsing file '" << *err.source().path
                  << "':\n" << err.description() << "\n (" << err.source().begin << ")\n";
        return 1;
    }

    /// Get parameters
    const auto use_gpu = ConfigGetValue<bool>(tbl, "OpenCL.use_gpu");
    const auto platform_id = ConfigGetValue<size_t>(tbl, "OpenCL.platform_id");
    const auto device_id = ConfigGetValue<size_t>(tbl, "OpenCL.device_id");

    const auto image_path = ConfigGetValue<std::string>(tbl, "Hough_transform.image");
    const auto hough_min_radius = ConfigGetValue<int>(tbl, "Hough_transform.min_radius");
    const auto hough_max_radius = ConfigGetValue<int>(tbl, "Hough_transform.max_radius");


    const auto device = GetDevice(platform_id, device_id, use_gpu);

    const cl::Context context(device);
    const cl::CommandQueue queue(context, device);

    const std::string kernelSource = ReadKernelFile("cl/FindCircle.cl");

    const cl::Program program(context, kernelSource);
    try {
        program.build({device});
    } catch (cl::Error& buildErr) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        throw;
    }


    // Load input image
    cv::Mat input_img = LoadInputImage(image_path);

    cv::Mat input_canny;
    cv::Canny(input_img, input_canny, 100, 200);
    cv::namedWindow("Canny", cv::WINDOW_NORMAL);
    cv::imshow("Canny", input_canny);

    cv::Mat circle_accumulator(input_img.rows, input_img.cols, CV_8UC1);

    // Create input/output of opencl images
    cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);

    cl::Image2D inputImageCL(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        format,
        input_canny.cols,
        input_canny.rows,
        0,
        input_canny.data
    );

    cl::Image2D output_find_circle_cl(
        context,
        CL_MEM_READ_WRITE,
        format,
        input_canny.cols,
        input_canny.rows
    );

    cl::Image2D output_find_radius_cl(
        context,
        CL_MEM_WRITE_ONLY,
        format,
        input_canny.cols,
        input_canny.rows
    );

    /// Kernel -> function name in program (.cl file)
    cl::Kernel kernel_find_circle(program, "FindCircle");
    cl::Kernel kernel_find_radius(program, "FindRadius");

    // Set kernel arguments
    kernel_find_circle.setArg(0, inputImageCL);
    kernel_find_circle.setArg(1, hough_min_radius);
    kernel_find_circle.setArg(2, hough_max_radius);
    kernel_find_circle.setArg(3, output_find_circle_cl);

    kernel_find_radius.setArg(0, output_find_circle_cl);
    kernel_find_radius.setArg(1, output_find_radius_cl);


    // Enqueue kernel execution
    cl::NDRange globalSize(input_canny.cols, input_canny.rows);

    queue.enqueueNDRangeKernel(kernel_find_circle, cl::NullRange, globalSize, cl::NullRange);
    queue.enqueueNDRangeKernel(kernel_find_radius, cl::NullRange, globalSize, cl::NullRange);

    // Create output image
    cv::Mat cv_output_find_circle(input_canny.rows, input_canny.cols, CV_8UC1);
    cv::Mat cv_output_find_radius(input_canny.rows, input_canny.cols, CV_8UC1);


    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region{};
    region[0] = static_cast<size_t>(input_canny.cols);
    region[1] = static_cast<size_t>(input_canny.rows);
    region[2] = 1;


    // Read back the result
    queue.enqueueReadImage( output_find_circle_cl, CL_TRUE, origin, region, 0, 0, cv_output_find_circle.data );
    queue.enqueueReadImage( output_find_radius_cl, CL_TRUE, origin, region, 0, 0, cv_output_find_radius.data );


    // Show result image
    ShowGrayscaleImage(cv_output_find_circle, "FindCircle");
    ShowGrayscaleImage(cv_output_find_radius, "FindRadius");

    cv::waitKey(0);

    return 0;
}
