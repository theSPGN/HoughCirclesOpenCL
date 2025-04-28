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

[[nodiscard]]
cl::Device GetDevice(bool useGPU)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        std::cerr << "No OpenCL platforms found.\n";
        throw std::runtime_error("No OpenCL platforms found.");
    }

    std::cout << "Found " << platforms.size() << " platform(s).\n";

    cl::Platform platform = platforms.front();
    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // Find devices
    std::vector<cl::Device> devices;
    cl_device_type deviceType = useGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    platform.getDevices(deviceType, &devices);

    if (devices.empty())
    {
        std::cerr << "No devices of type " << (useGPU ? "GPU" : "CPU") << " found on this platform.\n";
        throw std::runtime_error("No OpenCL device found.");
    }
    std::cout << "Found " << devices.size() << " device(s) of type " << (useGPU ? "GPU" : "CPU") << ".\n";

    // 4. Select the first device
    cl::Device device = devices.front();
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    return device;
};

cv::Mat LoadInputImage(const std::string &image_path)
{
    const cv::Mat inputImage = cv::imread(image_path, cv::IMREAD_COLOR_BGR);
    if (inputImage.empty())
    {
        throw std::runtime_error("Failed to load input.png");
    }
    return inputImage;
}

int main(int argc, char **argv)
{
    static const std::string image_path = "img/coins.png";
    const bool useGPU = true;

    const auto device = GetDevice(useGPU);

    const cl::Context context(device);
    const cl::CommandQueue queue(context, device);

    const std::string kernelSource = ReadKernelFile("cl/HoughTransformCircle_Kernels.cl");

    const cl::Program program(context, kernelSource);
    try {
        program.build({device});
    } catch (cl::Error& buildErr) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        throw;
    }

    cl::Kernel kernel(program, "hough_transform");

    // Load input image
    cv::Mat inputBGR = LoadInputImage(image_path);
    cv::Mat inputRGBA;
    cv::cvtColor(inputBGR, inputRGBA, cv::COLOR_BGR2RGBA);

    cv::namedWindow("input", cv::WINDOW_NORMAL);
    cv::imshow("input", inputBGR);

    // Create input/output of opencl images
    cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
    cl::Image2D inputImageCL(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        format,
        inputRGBA.cols,
        inputRGBA.rows,
        0,
        inputRGBA.data
    );

    cl::Image2D outputImageCL(
        context,
        CL_MEM_WRITE_ONLY,
        format,
        inputRGBA.cols,
        inputRGBA.rows
    );

    // Set kernel arguments
    kernel.setArg(0, inputImageCL);
    kernel.setArg(1, outputImageCL);

    // Enqueue kernel execution
    cl::NDRange globalSize(inputRGBA.cols, inputRGBA.rows);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);

    // Create output image
    cv::Mat outputRGBA(inputRGBA.rows, inputRGBA.cols, CV_8UC4);

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region{};
    region[0] = static_cast<size_t>(inputRGBA.cols);
    region[1] = static_cast<size_t>(inputRGBA.rows);
    region[2] = 1;

    // Read back the result
    queue.enqueueReadImage(
        outputImageCL,
        CL_TRUE,
        origin,
        region,
        0,
        0,
        outputRGBA.data
    );

    // Show result image
    cv::Mat outputBGR;
    cv::cvtColor(outputRGBA, outputBGR, cv::COLOR_RGBA2BGR);
    cv::normalize(outputBGR, outputBGR, 0, 255, cv::NORM_MINMAX);

    cv::namedWindow("OutputImage", cv::WINDOW_NORMAL);
    cv::imshow("OutputImage", outputBGR);
    cv::waitKey(0);

    return 0;
}
