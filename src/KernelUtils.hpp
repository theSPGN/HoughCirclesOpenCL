#ifndef KERNELUTILS_HPP
#define KERNELUTILS_HPP

#include <filesystem>
#include <string>
#include <iostream>
#include <toml++/toml.hpp>


[[nodiscard]]
std::string ReadKernelFile(const std::filesystem::path &path);


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


#endif //KERNELUTILS_HPP
