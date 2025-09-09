#include "Camera.h"
#include <iostream>  // For std::cout
#include <GLFW/glfw3.h>  // For GLFW functions and constants
#include <glm/gtc/matrix_transform.hpp>  // For glm::lookAt and glm::ortho

// Constructor
Camera::Camera()
    : m_Position(glm::vec3(0.0f, 0.0f, 3.0f)),
      m_Front(glm::vec3(0.0f, 0.0f, -1.0f)),
      m_Up(glm::vec3(0.0f, 1.0f, 0.0f)),
      m_Right(glm::vec3(1.0f, 0.0f, 0.0f)),
      m_WorldUp(glm::vec3(0.0f, 1.0f, 0.0f)),
      m_Orientation(glm::vec3(0.0f, 0.0f, -1.0f)),
      m_Yaw(-90.0f),
      m_Pitch(0.0f),
      m_MoveSpeed(2.5f),
      m_MouseSensitivity(0.1f),
      m_Zoom(45.0f),
      m_Left(-1.0f), m_Right1(1.0f), m_Bottom(-1.0f), m_Top(1.0f), m_Near(0.1f), m_Far(100.0f) {}

// Set Orthographic projection
void Camera::SetOrthographic(float near, float far)
{
    m_Near = near;
    m_Far = far;

    // Reset Projection and View matrices
    m_Projection = glm::ortho(m_Left, m_Right1, m_Bottom, m_Top, near, far);
    m_View = glm::lookAt(m_Position, m_Position + m_Orientation, m_Up);
}

// Update the camera's view matrix based on position, orientation


// Update the camera's projection matrix
glm::mat4 Camera::GetProjectionMatrix()
{
    return m_Projection;
}

// Input Callbacks
// void KeyCallback(GLFWwindow* window, int key, int scanCode, int action, int mods)
// {
//     Camera* camera = (Camera*) glfwGetWindowUserPointer(window);
//     if (!camera) {
//         std::cout << "Warning: Camera wasn't set as the Window User Pointer! KeyCallback is skipped" << std::endl;
//         return;
//     }

//     if (action == GLFW_PRESS || action == GLFW_REPEAT)
//     {
//         switch (key)
//         {
//             case GLFW_KEY_UP:
//                 std::cout << "UP Pressed" << std::endl;
//                 break;
//             case GLFW_KEY_DOWN:
//                 std::cout << "DOWN Pressed" << std::endl;
//                 break;
//             case GLFW_KEY_LEFT:
//                 std::cout << "LEFT Pressed" << std::endl;
//                 break;
//             case GLFW_KEY_RIGHT:
//                 std::cout << "RIGHT Pressed" << std::endl;
//                 break;
//             default:
//                 break;
//         }
//     }
// }

void MouseButtonCallback(GLFWwindow* window, double currMouseX, double currMouseY)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        std::cout << "MOUSE LEFT Click" << std::endl;
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        std::cout << "MOUSE RIGHT Click" << std::endl;
    }
}

void CursorPosCallback(GLFWwindow* window, double currMouseX, double currMouseY)
{
    Camera* camera = (Camera*) glfwGetWindowUserPointer(window);
    if (!camera) {
        std::cout << "Warning: Camera wasn't set as the Window User Pointer! KeyCallback is skipped" << std::endl;
        return;
    }

    camera->m_NewMouseX = camera->m_OldMouseX - currMouseX;
    camera->m_NewMouseY = camera->m_OldMouseY - currMouseY;
    camera->m_OldMouseX = currMouseX;
    camera->m_OldMouseY = currMouseY;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        std::cout << "MOUSE LEFT Motion" << std::endl;
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        std::cout << "MOUSE RIGHT Motion" << std::endl;
    }
}

// void ScrollCallback(GLFWwindow* window, double scrollOffsetX, double scrollOffsetY)
// {
//     Camera* camera = (Camera*) glfwGetWindowUserPointer(window);
//     if (!camera) {
//         std::cout << "Warning: Camera wasn't set as the Window User Pointer! ScrollCallback is skipped" << std::endl;
//         return;
//     }

//     std::cout << "SCROLL Motion" << std::endl;
// }

// Set up GLFW callbacks for input
void Camera::EnableInputs(GLFWwindow* window)
{
    // Set camera as the user pointer for the window
    glfwSetWindowUserPointer(window, this);

    // Handle key inputs
    // glfwSetKeyCallback(window, (void(*)(GLFWwindow *, int, int, int, int)) KeyCallback);

    // Handle cursor buttons
    glfwSetMouseButtonCallback(window, (void(*)(GLFWwindow *, int, int, int)) MouseButtonCallback);

    // Handle cursor position and inputs on motion
    glfwSetCursorPosCallback(window , (void(*)(GLFWwindow *, double, double)) CursorPosCallback);

    // Handle scroll inputs
    // glfwSetScrollCallback(window, (void(*)(GLFWwindow *, double, double)) ScrollCallback);
}
