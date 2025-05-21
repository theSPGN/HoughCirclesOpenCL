#ifndef KERNELUTILS_HPP
#define KERNELUTILS_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <filesystem>
#include <string>
#include <iostream>
#include <CL/cl.hpp>
#include <toml++/toml.hpp>


[[nodiscard]]
std::string ReadKernelFile(const std::filesystem::path &path);

[[nodiscard]]
cl::Device GetDevice(std::size_t platform_id, std::size_t device_id, bool use_gpu);

template <typename T>
[[nodiscard]]
T ConfigGetValue(const toml::table &tbl, const std::string_view key)
{
    auto opt_value = tbl.at_path(key).value<T>();

    if (!opt_value.has_value())
    {
        std::cerr << "Failed to get value from config file for key " << key << "\n";
        throw std::runtime_error("Failed to get value.");
    }
    return opt_value.value();
}

[[nodiscard]]
std::string_view GetOpenCLErrorString(cl_int error_code);

#endif //KERNELUTILS_HPP
