#version 450

//linked to the fragColor variable in vertexShader.vert 
//input variable here doesn't need to have the same name as the index will be used to link them
layout(location = 0) in vec3 fragColor; //input
layout(location = 0) out vec4 outColor; //output

void main() {
// Set the output color to red: RGB, alpha = 1 (fully opaque)
    outColor = vec4(fragColor, 1.0);
}