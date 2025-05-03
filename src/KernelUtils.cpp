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

