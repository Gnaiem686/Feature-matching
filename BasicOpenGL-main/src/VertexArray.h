#pragma once

#include <Debugger.h>
#include <VertexBuffer.h>
#include <VertexBufferLayout.h>

// VAO
class VertexArray
{
    private:
        unsigned int m_RendererID;
    public:
        VertexArray();
        ~VertexArray();
        
        void AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout);

        void Bind() const;
        void Unbind() const;
};