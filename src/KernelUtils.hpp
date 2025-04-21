#ifndef KERNELUTILS_HPP
#define KERNELUTILS_HPP

#include <filesystem>
#include <string>


[[nodiscard]]
std::string ReadKernelFile(const std::filesystem::path &path);


#endif //KERNELUTILS_HPP
