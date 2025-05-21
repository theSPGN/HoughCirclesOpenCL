#include "KernelUtils.hpp"

#include <fstream>
#include <sstream>


std::string ReadKernelFile(const std::filesystem::path &path)
{
    const std::ifstream file{path};

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file" + path.string());
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

[[nodiscard]]
cl::Device GetDevice(std::size_t platform_id, std::size_t device_id, bool use_gpu)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        std::cout << "No OpenCL platforms found.\n";
        throw std::runtime_error("No OpenCL platforms found.");
    }
    std::cout << "Found " << platforms.size() << " platform(s).\n";


    for (int i = 0; i < platforms.size(); ++i)
    {
        std::cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
    }
    if (platform_id >= platforms.size())
    {
        std::cout << "Platform with id=" << platform_id << " is out of range.\n";
        throw std::runtime_error("Non exsiting platform requested");
    }

    cl::Platform platform = platforms[platform_id];
    std::cout << "Using platform " << platform_id << ": " << platform.getInfo<CL_PLATFORM_NAME>() << "\n" << std::endl;

    // Find devices
    std::vector<cl::Device> devices;
    cl_device_type deviceType = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    try
    {
        platform.getDevices(deviceType, &devices);
    }
    catch (cl::Error& err)
    {
        std::cout << "OpenCL Error: " << err.what() << " (" << err.err() << ")" << std::endl;

        use_gpu = !use_gpu;
        deviceType = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
        std::cout << "Falling back to using " << (use_gpu ? "GPU" : "CPU")  << std::endl;
        try
        {
            platform.getDevices(deviceType, &devices);
        }
        catch (cl::Error& new_err)
        {
            std::cout << "OpenCL Error: " << new_err.what() << " (" << new_err.err() << ")" << std::endl;
            throw std::runtime_error("No available device with type cpu or gpu");
        }
    }

    if (devices.empty())
    {
        std::cout << "No devices of type " << (use_gpu ? "GPU" : "CPU") << " found on this platform.\n";
        throw std::runtime_error("No OpenCL device found.");
    }
    std::cout << "Found " << devices.size() << " device(s) of type " << (use_gpu ? "GPU" : "CPU") << ".\n";

    for (int i = 0; i < devices.size(); i++)
    {
        std::cout << "Device " << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    }
    if (device_id >= devices.size())
    {
        std::cout << "Device with id=" << device_id << " is out of range.\n";
        throw std::runtime_error("Non existing device requested");
    }
    cl::Device device = devices[device_id];
    std::cout << "Using device " << device_id << ": " << device.getInfo<CL_DEVICE_NAME>() << "\n" << std::endl;;

    return device;
};


std::string_view GetOpenCLErrorString(cl_int error_code)
{
    switch (error_code)
    {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        default: return "Unknown OpenCL error";
    }
}

