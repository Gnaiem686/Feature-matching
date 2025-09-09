#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <Texture.h>

Texture::Texture(const std::string& filepath)
    : m_RendererID(0), m_Filepath(filepath), m_LocalBuffer(nullptr), m_Width(0), m_Height(0), m_Components(0)
{
    // Flips the image so it appears right side up
    stbi_set_flip_vertically_on_load(1);

    // Reads the image from a file and stores it in m_LocalBuffer
    m_LocalBuffer = stbi_load(filepath.c_str(), &m_Width, &m_Height, &m_Components, 4);

    // Generates an OpenGL texture object
    GLCall(glGenTextures(1, &m_RendererID));

    // Assigns the texture to a Texture Unit
    GLCall(glBindTexture(GL_TEXTURE_2D, m_RendererID));

    // Configures the type of algorithm that is used to make the image smaller or bigger
    GLCall(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR));
    GLCall(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

    // Configures the way the texture repeats (if it does at all)
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

    // Assigns the image to the OpenGL Texture object
    GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_LocalBuffer));

    // Generates Mipmaps
	GLCall(glGenerateMipmap(GL_TEXTURE_2D));

    // Unbinds the OpenGL Texture object so that it can't accidentally be modified
    GLCall(glBindTexture(GL_TEXTURE_2D, 0));

    if (m_LocalBuffer)
    {
        // Deletes the image data as it is already in the OpenGL Texture object
        stbi_image_free(m_LocalBuffer);
    }
}

Texture::~Texture()
{
    GLCall(glDeleteTextures(1, &m_RendererID));
}

void Texture::Bind(unsigned int slot) const
{
    GLCall(glActiveTexture(GL_TEXTURE0 + slot));
    GLCall(glBindTexture(GL_TEXTURE_2D, m_RendererID));
}

void Texture::Unbind() const
{
    GLCall(glBindTexture(GL_TEXTURE_2D, 0));
}