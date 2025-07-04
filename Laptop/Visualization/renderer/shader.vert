#version 450

// Bind the model view projection object
// Note that word-alignment requirements for UBO matters, so mat4 = vec4 = 4N (N = bytes in each element, e.g. 4 or 8) byte alignment, while vec 2 is 8 byte alignment, and so on.
// To avoid this, the alignas(X-multiple) glm::mat4: model could be used. 
// Alternatively do #define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES in the hellotriangle.cpp, but it sometimes breaks down with nested structures
// -> A structure must be aligned to 16 byte multiples; it's the largest of any of its members (1,2,4,8 -> 16 byte alignment).
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Assign vertex attributes to specific locations for later access
layout(location = 0) in vec3 inPosition; // vertex attribute: x,y,z
layout(location = 1) in vec3 inColor;  // vertex attribute: r,g,b
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

//main: Invoked for every vertex
void main() {
    //The built-in gl_VertexIndex variable contains the index of the current vertex
    //currently contains dummy z and w values
     gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}