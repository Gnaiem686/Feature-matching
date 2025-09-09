#pragma once

#include <glad/glad.h>

#include <iostream>

// #define ASSERT(x) if (!(x)) __debugbreak();

#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h> // For __debugbreak() on Windows
#define ASSERT(x) if (!(x)) __debugbreak();
#elif defined(__APPLE__) || defined(__MACH__)
#include <signal.h> // For raising SIGTRAP on macOS
#include <unistd.h>
#define ASSERT(x) if (!(x)) raise(SIGTRAP);
#else
#include <signal.h> // For other Unix-like systems
#define ASSERT(x) if (!(x)) raise(SIGTRAP);
#endif

#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x, __FILE__, __LINE__));

void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);