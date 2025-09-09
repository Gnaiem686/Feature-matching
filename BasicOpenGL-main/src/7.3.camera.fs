#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

uniform vec3 color;      // Used if useUniformColor is true
uniform bool useUniformColor;

void main()
{
    if (useUniformColor)
        FragColor = vec4(color, 1.0);
    else
        FragColor = vec4(vertexColor, 1.0);
}
