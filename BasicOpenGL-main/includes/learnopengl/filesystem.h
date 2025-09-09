#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <cstdlib>

class FileSystem
{
private:
    typedef std::string (*Builder)(const std::string& path);

public:
    static std::string getPath(const std::string& path)
    {
        static Builder pathBuilder = getPathBuilder();
        return (*pathBuilder)(path);
    }

private:
    static std::string const& getRoot()
    {
        // Allow user to override via environment variable
        static const char* envRoot = std::getenv("LOGL_ROOT_PATH");

        // Fallback: hardcoded base path (e.g., project root relative to binary)
        static const std::string defaultRoot = ""; // <-- You can hardcode a path here
        static const std::string root = (envRoot != nullptr) ? std::string(envRoot) : defaultRoot;

        return root;
    }

    static Builder getPathBuilder()
    {
        if (!getRoot().empty())
            return &FileSystem::getPathRelativeRoot;
        else
            return &FileSystem::getPathRelativeBinary;
    }

    static std::string getPathRelativeRoot(const std::string& path)
    {
        return getRoot() + "/" + path;
    }

    static std::string getPathRelativeBinary(const std::string& path)
    {
        return "../../../" + path; // relative to executable
    }
};

#endif // FILESYSTEM_H
