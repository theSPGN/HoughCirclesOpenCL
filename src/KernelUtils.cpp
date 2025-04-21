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
