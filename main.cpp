#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "KernelUtils.hpp"


int main(int argc, char **argv)
{
    cv::namedWindow("Circles", cv::WINDOW_AUTOSIZE);
    const cv::Mat image = cv::imread("data/circuit.bmp", cv::IMREAD_GRAYSCALE);

    cv::imshow("Circles", image);
    cv::waitKey();

    cl_uint num_platforms;
    cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);

    if (status != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to get the number of platforms" << std::endl;
        return EXIT_FAILURE;
    }

    if (num_platforms == 0)
    {
        std::cerr << "Error: No platforms found" << std::endl;
        return EXIT_FAILURE;
    }

    cl_platform_id platform = [&]() {
        std::vector<cl_platform_id> platforms(num_platforms);
        status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        return platforms[0];
    }();

    cl_uint num_devices = 0;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    std::vector<cl_device_id> devices(num_devices);

    if (num_devices == 0)
    {
        std::cout << "No GPU device available.\n";
        std::cout << "Choose CPU as default device." << std::endl;

        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
    }
    else
    {
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    }

    cl_context context = clCreateContext(nullptr, 1, devices.data(), nullptr, nullptr, &status);

    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &status);



    const std::filesystem::path kernel_path = "cl/Kernel.cl";
    const std::string kernel_code = ReadKernelFile(kernel_path);
    const char *c_kernel_code = kernel_code.c_str();
    std::size_t kernel_size[] = {kernel_code.size()};

    cl_program program = clCreateProgramWithSource(context, 1, &c_kernel_code, kernel_size, &status);

    clBuildProgram(program, 1, devices.data(), nullptr, nullptr, nullptr);



	std::string input = "GdkknVnqkc";
    std::vector<char> output(input.size() + 1, '\0');

    cl_mem input_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(char), input.data(), &status);
    cl_mem output_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(char), nullptr, &status);

    cl_kernel kernel = clCreateKernel(program, "helloworld", nullptr);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_mem);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_mem);

    std::size_t global_work_size[] = {input.size()};
    status = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);


    status = clEnqueueReadBuffer(command_queue, output_mem, CL_TRUE, 0, input.size(), output.data(), 0, nullptr, nullptr);


    const std::string output_str(output.begin(), output.end());
    std::cout << output_str << std::endl;

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(input_mem);
    clReleaseMemObject(output_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);


    return EXIT_SUCCESS;
}