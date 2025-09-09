#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Camera {
public:
    Camera();

    void SetOrthographic(float near, float far);
    glm::mat4 GetViewMatrix();
    glm::mat4 GetProjectionMatrix();
    void EnableInputs(GLFWwindow* window);

    // Camera properties
    glm::vec3 m_Position;
    glm::vec3 m_Front;
    glm::vec3 m_Up;
    glm::vec3 m_Right;
    glm::vec3 m_WorldUp;
    glm::vec3 m_Orientation;

    // Projection properties
    float m_Left, m_Right1, m_Bottom, m_Top, m_Near, m_Far;

    // Mouse control variables
    double m_OldMouseX, m_OldMouseY, m_NewMouseX, m_NewMouseY;

    // Other properties
    float m_Yaw, m_Pitch;
    float m_MoveSpeed;
    float m_MouseSensitivity;
    float m_Zoom;

    glm::mat4 m_Projection;
    glm::mat4 m_View;
};

#endif // CAMERA_H
