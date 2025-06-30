#version 450

// Bind the model view projection object
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Assign vertex attributes to specific locations for later access
layout(location = 0) in vec2 inPosition; // vertex attribute: x,y 
layout(location = 1) in vec3 inColor;  // vertex attribute: r,g,b

layout(location = 0) out vec3 fragColor;

//main: Invoked for every vertex
void main() {
    //The built-in gl_VertexIndex variable contains the index of the current vertex
    //currently contains dummy z and w values
     gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}