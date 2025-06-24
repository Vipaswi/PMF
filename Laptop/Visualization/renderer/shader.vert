#version 450

layout(location = 0) out vec3 fragColor;

//Mimiced vertex buffer data for a triangle
vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

//Color data for each vertex
vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

//main: Invoked for every vertex
void main() {
    //The built-in gl_VertexIndex variable contains the index of the current vertex
    //currently contains dummy z and w values
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}