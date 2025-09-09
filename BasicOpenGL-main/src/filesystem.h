// filesystem.h
#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <cstdlib>

class FileSystem
{
public:
    static std::string getPath(const std::string& relativePath)
    {
#ifdef _WIN32
        std::string basePath = "./";
#else
        std::string basePath = "./";
#endif
        return basePath + relativePath;
    }
};

#endif
