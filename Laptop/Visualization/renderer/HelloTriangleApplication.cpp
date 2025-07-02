#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h> // GLFW library for window management and input handling

#include <iostream>  // reporting errors
#include <stdexcept> // propagating errors
#include <cstdlib>   // EXIT_SUCCESS, EXIT_FAILURE
#include <vector>
#include <optional> //for queue family indices: to determine queue nonexistence
#include <set>

#include <cstdint> // uint32_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::clamp

#include <fstream> // for file reading

#include <glm/glm.hpp> //Linear algebra library for graphics software
#include <array>

#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp> // Allows model transformations e.g. glm:rotate, lookAt, perspective

#include <chrono> // Ensures precise timekeeping (to make rendering independent of framerate)

// Image use:
#define STB_IMAGE_IMPLEMENTATION // Includes function bodies for header file
#include <stb_image.h> 


const uint32_t WIDTH = 800; // width of the window
const uint32_t HEIGHT = 600; // height of the window

//Validation Layers: Currently the standard validation layer provided by Khronos.
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//if not debug mode, disable validation
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

/// <summary>
/// Determine whether the instance has the debugUtilsMessenger extension available, and if so, create it and return the VKResult.
/// </summary>
/// <param name="instance"> the VKInstance to make the messenger for</param>
/// <param name="pCreateInfo"> the creation information </param>
/// <param name="pAllocator"> the allocator, if any</param>
/// <param name="pDebugMessenger"> the debug message</param>
/// <returns>The VKResult of the function creation call or an error indicating the function wasn't present</returns>
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

/// <summary>
/// Destroys the debug utils messenger created by CreateDebugUtilsMessengerEXT.
/// </summary>
/// <param name="instance"> The instance to which the messenger belonged</param>
/// <param name="debugMessenger"> The messenger to be destroyed</param>
/// <param name="pAllocator"> The allocator, if any</param>
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}


/// <summary>
/// The main application, whose private members and functions will interact/store Vulakn Objects.
/// </summary>
class HelloTriangleApplication {

public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    static std::vector<char> readFile(const std::string& filename) {
        //flags: ate: start reading from end of file; binary: read as binary file
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!" + filename);
        }

        //get the size of the file through the current position of the file pointer (.tellg()) (which is why we used ate)
        //create a standard character vector of that size
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

		//go back to the start, and read the file into the buffer
        file.seekg(0);
        file.read(buffer.data(), fileSize);

		//close the file
        file.close();

        //return file character data
        return buffer;
    }

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };
    
    struct Vertex {
		glm::vec2 pos;
        glm::vec3 color;
        glm::vec2 texCoord;

        // Defines the input-rate and size of vertex data
        static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription bindingDescription{};
            bindingDescription.binding = 0; // Binding index in binding arr.
            bindingDescription.stride = sizeof(Vertex); // # bytes bet entries
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;  // When to move data; current: after each vertex

            return bindingDescription;
        };

        // How to extract vertex attributes from chunks of vertex data from binding descriptions
        static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
            std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
            // The binding the per vertex data originates from
            attributeDescriptions[0].binding = 0; 
            // location directive in the vertex shader input
            attributeDescriptions[0].location = 0; 
            // Attribute data type:
			    // -> weirdly enough, we use the same formats as colors for vector types (e.g. vec2 = R32G32)
                // Also silently defines byte size of attribute data
            attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; 
            // Offset from per-vertex start data to access the param.
            attributeDescriptions[0].offset = offsetof(Vertex, pos); 

            // Similarly: 
            attributeDescriptions[1].binding = 0; 
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[1].offset = offsetof(Vertex, color);

            attributeDescriptions[2].binding = 0;
            attributeDescriptions[2].location = 2;
            attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
            attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
            return attributeDescriptions;
        };
    };

    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; //implicitly destroyed btw
    VkDevice device; //a logical device that interfaces with the physical device
    VkQueue graphicsQueue; //handle to interface with graphics queue, implicitly cleaned up when the device is destroyed
    VkQueue presentQueue; 
	VkSurfaceKHR surface; //the surface to render to; requires the VK_KHR_win32_surface extension for Windows, added by glfw's required extensions
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages; //handles to swapchain images; automatically cleaned by swap chain
    std::vector<VkImageView> swapChainImageViews; //image views describes image access 
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderPass;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool; //Memory management for command buffers
	std::vector<VkCommandBuffer> commandBuffers; //automatically freed on command pool destruction
    
    // Synchronization: 
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    // Get more frames 
    const int MAX_FRAMES_IN_FLIGHT = 2;
    uint32_t currentFrame = 0;

    // Resizing:
    bool framebufferResized = false;
    
    // Vectors: position, color
    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
    };
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    // Index Buffer:
    const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    // Uniform Buffer Object
    struct UniformBufferObject {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
    };

    VkDescriptorSetLayout descriptorSetLayout;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    // Images:
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    void initWindow() {
		glfwInit(); // Initialize GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Specify that we don't want an OpenGL context
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Make the window not resizable for now.

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Width, height, title, monitor to open the window on, and OPENGL context to share with other windows (nullptr for no sharing).
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

   void initVulkan() {
		createInstance(); // Create a Vulkan instance
		setupDebugMessenger(); // Setup the debug messenger for validation layers
        createSurface();
		pickPhysicalDevice();
		createLogicalDevice(); 
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
		createGraphicsPipeline(); 
		createFramebuffers(); 
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void cleanup() {
        cleanupSwapChain();

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    /** Texture sampler that can be applied to any image(1D, 2D, or 3D).
    *   
    *   It provides an interface to extract colors from a texture
    */
    void createTextureSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        // How to interpolate magnified or minified texels. Alt. NEAREST
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        // Addressing mode per axis (x,y,z) = (u,v,w) for texture spaces
            // -> Repeat repeats the texture when extending beyond image dimension
            // -> Won't really be used for this tutorial (staying within dimensions), but for ceilings/floor textures, repeat is helpful.
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        // Enable Anisotropy
        samplerInfo.anisotropyEnable = VK_TRUE;
            // Determine max anisotropy of physical device:
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

        // Border color
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

        // Determine coordinate system. VK_FALSE means [0,1) coordinate sys.
        // -> Real world coordinates use normalized ranges to use a variety of resolutions with the same coords.
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        // Disable texel comparison
        // -> "Mainly used for percentage-closer filtering on shadow maps"
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    
        // Mipmapping filter:
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        // Create Sampler:
        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        // If transferring queue family ownership, these need to be filled
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // The image that is affected and the specific parts of th eimage:
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        // DO after we figure out the transitions to use
        barrier.srcAccessMask = 0; // TODO
        barrier.dstAccessMask = 0; // TODO

        // Pipeline stage Flags
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        // If we're just writing, complete in the earliest stage
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;   // Earliest stage
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // Pseudo stage to complete by/in
        }
        // Complete transfer prior to fragment shader
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }


        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage, // Stage prior to barrier and operations that wait on the barrier
            0,          // VK_DEPENDENCY_BY_REGION alt. : allows reading parts that have been written to
            0, nullptr, // Memory Barriers
            0, nullptr, // Image Barriers
            1, &barrier // Image Memory Barriers
        );

        endSingleTimeCommands(commandBuffer);
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, 
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {

        // Create Image Object
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        // 1D: Array of data/gradient
        // 2D: textures (mainly)
        // 3D: e.g. voxel volumes
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1; // 2D image has to have depth of one
        imageInfo.mipLevels = 1; //no mipMapping (having 1+ image levels to use smaller images when they're farther away)
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling; //LINEAR (row order) or OPTIMAL (implementation defined order)
        // Initially not usable by GPU, but undefined discards texels while PREINITIALIZED preserves on first transition:
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Only one queue family uses this (graphics and transfer op)
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; // Multisampling
        imageInfo.flags = 0; // For sparse images, there are flags (i.e. images where only certain regions backed by memory, like having a bunch of air)

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        // Now allocate memory for the image:
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void createTextureImage() {

        // Load an image with rgb and alpha channels. Also, fill in the width, height, and actual channels it has
        int texWidth, texHeight, texChannels;

        stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); 
        VkDeviceSize imageSize = texWidth * texHeight * 4; //four bytes per pixel with rgba.

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        // Create buffer in host visible memory to vkMapMemory the contents 
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, imageSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // Clean pixel array:
        stbi_image_free(pixels);

        // Image used as the dst from the buffer copy, and read by a shader through a sampler (an object that tells the GPU how to read textures) for usage
        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        // Transition the image layout from undefined to optimal
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Cleanup
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        // Configure Descriptor Sets; descriptors for buffers are configured with VkDescriptorBufferInfo
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject); // One ubo, or VK_WHOLE_RANGE

            // Specify Image descriptor information
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            // Combine into an array of WriteDescriptorSets, required for updating descriptor sets
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            // Update
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    // Decribes the type of resources bound to so the vertex shader can access those resources
    void createDescriptorSetLayout() {
        // Bind to the vertex shader binding index 0 with type Uniform buffer
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; 
        uboLayoutBinding.pImmutableSamplers = nullptr; // Only rel. for image sampling descriptors

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            // Get a pointer where we can write data to later on: persistent mapping
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer{};
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        // Query available memory types:
        // -> has memoryTypes and memoryHeaps arrays
        // -> memoryTypes: VkMemoryType structs with heap and memory properties
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
       
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    // Creates a buffer with the fiven param info. 
        // note: In Real Life Applications, memory allocation should be handled with a allocator that splits
        //       a single allocation among many objects using the offset params (as the max mem. allocation count can be p. low). 
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Only used by graphics queue

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create vertex buffer!");
        }

        // Size (may differ from abv.), alignment (depends on usage and flag), memtype (compatible mem_types)
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory");
        }

        // Fourth param is offset in memory region. If non-zero (not only for 
        // this one buffer, then it has to be divisible by memRequirements.alignment
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        //Create a staging buffer to be used as a source in memory transfer operations
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);


        // Memory access
            // -> Note that drivers may not immediately copy the data into buffer memory due to caching
            // -> but this has been avoided by making the memory host (cpu) coherent (synced).
            // -> otherwise explicit flushing and invalidating (vkFlushMappedMemoryRanges, vkInvalidateMappedMemoryRanges) would be 
            // -> used, which generally offers better performance.
                // -> Mapped memory is currently coherent with allocated memory

        // Copy data into the staging buffer:
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        // Copy staging buffer contents
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        // Cleanup:
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void cleanupSwapChain() {
        for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
        
    }

    // Optimally, instead, you would call VkSwapChainCreateInfo with the new swap chain
    // and the old swap chain. 
    void recreateSwapChain() {

		// If it's 0, the window is minimized, so we don't recreate the swap chain
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        //Recreate the swap chain, image views, and frame buffers
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        // Simple rotation around the z axis using the time variable:
        UniformBufferObject ubo{};
            // -> Takes the existing transofrmation, rotation angle, and rotation axis as params
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        // View transofrmation to look at the geometry from 45 deg. above
            // -> params: eye position, center position, and up axis (2,2,2 makes equilateral triangle so deg. is 45)
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        // Projection transformation with 45 deg. FOV
            // -> params: FOV angle, aspect ratio, near and far view planes
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

        // Invert Y coordinate as GLm was designed for OpenGL.
        ubo.proj[1][1] *= -1;

        // Now copy data from uniform buffer object to the uniform buffer; better to use push constants tho :(
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        // Wait for previous frame to finish
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        updateUniformBuffer(currentFrame);

        // Acquire Swap Chain Image
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Only reset the fence if we are submitting work
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Record command buffer that draws a scene onto the retrieved image
        vkResetCommandBuffer(commandBuffers[currentFrame], 0); // Second param is a flag
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // Submit recorded command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            // Wait stages rely on wait semaphores (index matches) to begin
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

            // The command buffers to actually execute
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];


            // The semaphores to signal once the command buffers finish exec.
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // Present Swap chain image
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr; // allows u to check if the presentation was successful for all swap chains, but not needed here

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // One time submission command buffer allocation and start
    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // Usage of this specific command buffer info:
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        // Black
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; //primary command buffer, which can be submitted to a queue
        allocInfo.commandBufferCount = 1;

        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
            swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};

		// Match colorAttachment format to the swap chain image format
		colorAttachment.format = swapChainImageFormat; 
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; //no multisampling for now

        // What to do with the data prior to and after rendering
        
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;   // : Clear values to a constant at start (black)
	        // Other options include:
                // -> DONT_CARE: undefined, and we don't care
		        // -> preserve existing attachment content
        
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // : stored and available for reading
		    // -> DONT_CARE: undefined, and we don't care

        // Similar for stencil data : no stencil buffer so irrelavent
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // VKImage objects also represent textures and framebuffers, but pixel layout
        // in memory can differ based on purpose.
			// -> Layout_COLOR_ATTACHMNET_OPTIMAL: optimal for rendering to the image
		    // -> Layout_PRESENT_SRC_KHR: optimal for presenting the image to the screen
			// -> Layout_TRANSFER_SRC_OPTIMAL: optimal for copying from the image
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; 

        VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0; //index of the attachment in the render pass
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // Refers to implicit subpasses prior to and after a subpass
        dependency.dstSubpass = 0; // The only subpass; dst > src unlss src = vk_subpass_external
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; //stage to wait on 
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		// The dependency ensures that the color attachment is written to before the subpass begins
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

    }

    void createGraphicsPipeline() {
	   //read vertex and fragment shader code from SPIR-V files
       auto vertShaderCode = readFile("shaders/vert.spv");
       auto fragShaderCode = readFile("shaders/frag.spv");

	   //allowed to destroy shader modules after pipeline creation
       VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
       VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);


	   //vertex shader stage info
       VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
       vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	   vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; //describes it's the vertex shader stage

	   vertShaderStageInfo.module = vertShaderModule; //the shader module to use
	   vertShaderStageInfo.pName = "main";            //the shader entry point; main as default. i.e. shader modules can be combined and diff. entry points can be used for different purposes

       //pSpecializationInfo is used to specify values for shader constants
       //which makes compilation faster as it eliminates the need for variables at render time
	   //for now, we don't use it

	   //fragment shader stage info
       VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
       fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
       fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
       fragShaderStageInfo.module = fragShaderModule;
       fragShaderStageInfo.pName = "main";

       VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	   //Dynamic state is used to change certain pipeline states at runtime without recreating the pipeline
       std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
       };

       VkPipelineDynamicStateCreateInfo dynamicState{};
       dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
       dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
       dynamicState.pDynamicStates = dynamicStates.data();

       VkPipelineViewportStateCreateInfo viewportState{};
       viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
       viewportState.viewportCount = 1;
       viewportState.scissorCount = 1;

       //describes formatting of vertex input data 
       VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	   vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

       auto bindingDescription = Vertex::getBindingDescription();
       auto attributeDescriptions = Vertex::getAttributeDescriptions();

       //describes it in terms of bindings:
            //i.e. data spacing, and whether data is per-vertex or per-instance
	   vertexInputInfo.vertexBindingDescriptionCount = 1; //Zero as currently no vertex input data is used currently
       vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional

	   //and attributes: (e.g. position, color, texture coordinates)
       vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()); 
       vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional

       //Describes the type of geometry to draw from vertices, and primitive restart mode (on/off)
			//-> geometry types defined by VkPrimitiveTopology enum
	   VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
       inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
       inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; 
	   inputAssembly.primitiveRestartEnable = VK_FALSE; //don't need to break up lines and triangles with a special index value

	   //The region of the framebuffer to draw to from 0,0 to width,height
			//-> Viewports define the region of the framebuffer to draw to
			//-> scissors define the regions where pixels will actually be stored
				//-> Common for both of these to be dynamic for flexibility, with no performance penalty
       VkViewport viewport{};
       viewport.x = 0.0f;
       viewport.y = 0.0f;
       viewport.width = (float)swapChainExtent.width;
       viewport.height = (float)swapChainExtent.height;
       viewport.minDepth = 0.0f; //standard
	   viewport.maxDepth = 1.0f; //standard
        
       //draw to the entire framebuffer
       VkRect2D scissor{};
       scissor.offset = { 0, 0 };
       scissor.extent = swapChainExtent;

       //The creation of the rasterizer stage, converting vertex data to fragments
       VkPipelineRasterizationStateCreateInfo rasterizer{};
       rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	   rasterizer.depthClampEnable = VK_FALSE; //don't clamp depth values to the range [0,1]; instead, discard fragments outside the range
       rasterizer.rasterizerDiscardEnable = VK_FALSE; //draw fragments, don't discard :(
	   rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill the triangles, not wireframe or point
       rasterizer.lineWidth = 1.0f; //line thickness
       rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; //type of face culling
	   rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // clockwise to avoid backface culling due to Y-flip in matrix transforms (UBO)
       rasterizer.depthBiasEnable = VK_FALSE;
       rasterizer.depthBiasConstantFactor = 0.0f; // Optional
       rasterizer.depthBiasClamp = 0.0f; // Optional
       rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

       //multisampling, currently disabled
       VkPipelineMultisampleStateCreateInfo multisampling{};
       multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
       multisampling.sampleShadingEnable = VK_FALSE;
       multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
       multisampling.minSampleShading = 1.0f; // Optional
       multisampling.pSampleMask = nullptr; // Optional
       multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
       multisampling.alphaToOneEnable = VK_FALSE; // Optional

       //Color blending:
       VkPipelineColorBlendAttachmentState colorBlendAttachment{};
       colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
       colorBlendAttachment.blendEnable = VK_FALSE;
       colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
       colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
       colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
       colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
       colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
       colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

	   //color blending state, which describes how the color of the fragment is blended with the color already in the framebuffer
       VkPipelineColorBlendStateCreateInfo colorBlending{};
       colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
       colorBlending.logicOpEnable = VK_FALSE;
       colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
       colorBlending.attachmentCount = 1;
       colorBlending.pAttachments = &colorBlendAttachment;
       colorBlending.blendConstants[0] = 0.0f; // Optional
       colorBlending.blendConstants[1] = 0.0f; // Optional
       colorBlending.blendConstants[2] = 0.0f; // Optional
       colorBlending.blendConstants[3] = 0.0f; // Optional

       VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
       pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
       pipelineLayoutInfo.setLayoutCount = 1; 
       pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; 
       pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
       pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

       if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
           throw std::runtime_error("failed to create pipeline layout!");
       }

       VkGraphicsPipelineCreateInfo pipelineInfo{};
       pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
       pipelineInfo.stageCount = 2;
       pipelineInfo.pStages = shaderStages;

       pipelineInfo.pVertexInputState = &vertexInputInfo;
       pipelineInfo.pInputAssemblyState = &inputAssembly;
       pipelineInfo.pViewportState = &viewportState;
       pipelineInfo.pRasterizationState = &rasterizer;
       pipelineInfo.pMultisampleState = &multisampling;
       pipelineInfo.pDepthStencilState = nullptr; // Optional
       pipelineInfo.pColorBlendState = &colorBlending;
       pipelineInfo.pDynamicState = &dynamicState;

       pipelineInfo.layout = pipelineLayout;
       pipelineInfo.renderPass = renderPass;
       pipelineInfo.subpass = 0;

       pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
       pipelineInfo.basePipelineIndex = -1; // Optional

       if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
           throw std::runtime_error("failed to create graphics pipeline!");
	   }

	   //destroy created shader modules
       vkDestroyShaderModule(device, fragShaderModule, nullptr);
       vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    /**
		creates a shader module from the given SPIR-V code.
    */
   VkShaderModule createShaderModule(const std::vector<char>& code) {
       VkShaderModuleCreateInfo createInfo{};
       createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
       createInfo.codeSize = code.size(); //size of the code in bytes
       //.data() returns a pointer to the first element of the vector, a char array
	   //the reinterpret_cast is used to convert the char pointer to a uint32_t pointer, as required by pCode 
       //Usually reinterpreting would be dangerous as characters are byte aligned and uint32_t is 4 bytes aligned, but std::vector guarantees data alignment
       createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); 

       VkShaderModule shaderModule;
       if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
           throw std::runtime_error("failed to create shader module!");
       }

	   return shaderModule; //return the created shader module
   }

   VkImageView createImageView(VkImage image, VkFormat format) {
       VkImageViewCreateInfo viewInfo{};
       viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
       viewInfo.image = image;
       viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
       viewInfo.format = format;
       viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
       viewInfo.subresourceRange.baseMipLevel = 0;
       viewInfo.subresourceRange.levelCount = 1;
       viewInfo.subresourceRange.baseArrayLayer = 0;
       viewInfo.subresourceRange.layerCount = 1;

       VkImageView imageView;

       if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
           throw std::runtime_error("failed to create image view!");
       }

       return imageView;
   }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size()); 
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
        }

    }

   void createSwapChain() {
       SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

       VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
       VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
       VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

       uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

	   //if there is a maximum (not zero) and the image count exceeds it, set the image count to the maximum
       if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
           imageCount = swapChainSupport.capabilities.maxImageCount;
       }

       VkSwapchainCreateInfoKHR createInfo{};
       createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
       createInfo.surface = surface; //the surface to render to

       createInfo.minImageCount = imageCount;
       createInfo.imageFormat = surfaceFormat.format;
       createInfo.imageColorSpace = surfaceFormat.colorSpace;
       createInfo.imageExtent = extent;
       createInfo.imageArrayLayers = 1; //layers in each image
       createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; //VK_IMAGE_USAGE_TRANSFER_DST_BIT for post processing; currently directly rendered (no operations!).

       //Determine if we're using the same queue family for graphics and presentation or not
       QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
       uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

       //draw in graphics queue, present in present queue

       //If the queue familes differ, then it's best to stick with concurrent mode to avoid ownership transfer in exclusive mode (VK_SHARING_MODE_EXCLUSIVE/CONCURRENT); exclusive: better performance
       //Concurrent mode requires that you specify among the queue family indices that the image will be shared among
       if (indices.graphicsFamily != indices.presentFamily) {
           createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
           createInfo.queueFamilyIndexCount = 2;
           createInfo.pQueueFamilyIndices = queueFamilyIndices;
       }
       else {
           createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
           createInfo.queueFamilyIndexCount = 0; // Optional
           createInfo.pQueueFamilyIndices = nullptr; // Optional
       }

	   createInfo.preTransform = swapChainSupport.capabilities.currentTransform; //keep current transform; alternatively supportedTransforms can be queried to determine which transforms are supported, and used here
	   createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; //ignore alpha channel in most cases; like opacity of the window
       createInfo.presentMode = presentMode;
	   createInfo.clipped = VK_TRUE; //don't care about obscured pixels (e.g. by other windows)
       createInfo.oldSwapchain = VK_NULL_HANDLE; //no old swapchain for now; reference to old swapchain required if window resize occurs

       if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
           throw std::runtime_error("failed to create swap chain!");
       }


       vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
       swapChainImages.resize(imageCount);
       vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
       swapChainImageFormat = surfaceFormat.format;
       swapChainExtent = extent;
   }

   void createSurface() {
       if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
		   throw std::runtime_error("failed to create window surface!");
       }
   }

   void createLogicalDevice() {
       //get the QueueFamilyIndeices
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        //create a vector of queueCreateInfos for each QueueFamily
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        //for each unique queue family, define the following in its create info
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        //for the device handle, get the device queues
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }


   bool isDeviceSuitable(VkPhysicalDevice device) {
	   QueueFamilyIndices indices = findQueueFamilies(device);

       bool extensionsSupported = checkDeviceExtensionSupport(device);

       bool swapChainAdequate = false;
       if (extensionsSupported) {
           SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
           swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
       }

       VkPhysicalDeviceFeatures supportedFeatures;
       vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

       return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
   }

   bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
       uint32_t extensionCount = 0;
       vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

       std::vector<VkExtensionProperties> availableExtensions(extensionCount);
       vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

       std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

       for (const auto& extension : availableExtensions) {
           requiredExtensions.erase(extension.extensionName);
       }

       return requiredExtensions.empty();

       
   }

   void pickPhysicalDevice() {
       //alternatively, a new scoring function can be used to choose the best device for the use case
       //but for now, we pick any suitable device
       uint32_t deviceCount = 0;
       vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

       if (deviceCount == 0) {
           throw std::runtime_error("Failed to find GPUs that support Vulkan :(");
       }

       std::vector<VkPhysicalDevice> devices(deviceCount);
       vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

       for (const auto& device : devices) {
           if (isDeviceSuitable(device)) {
               physicalDevice = device;
               break;
           }
       }

       if (physicalDevice == VK_NULL_HANDLE) {
           throw std::runtime_error("failed to find a suitable GPU!");
       }
   }

   QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        //find graphics queue family
        QueueFamilyIndices indices; 

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        int i = 0; //index to assign to queue families
        for (const auto& queueFamily : queueFamilies) {
            VkBool32 presentSupport = false;
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            }
            if (presentSupport) {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }

            i++;
        }
        return indices;
   }


   void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
       createInfo = {};
       createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
       createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
       createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
       createInfo.pfnUserCallback = debugCallback;
   }

   void setupDebugMessenger() {
       if (!enableValidationLayers) return;

       VkDebugUtilsMessengerCreateInfoEXT createInfo;
       populateDebugMessengerCreateInfo(createInfo);

       if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
           throw std::runtime_error("failed to set up debug messenger!");
       }
   }

   void createInstance() {
       if (enableValidationLayers && !checkValidationLayerSupport()) {
           throw std::runtime_error("validation layers requested, but not available!");
       }

       //Optional but highly recomended:
       VkApplicationInfo appInfo{}; //declaration with zero-initialization
	   appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // Set the type of the structure
	   appInfo.pApplicationName = "Hello Triangle"; // Set the name of the application
       appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); 
       appInfo.pEngineName = "No Engine";
       appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
       appInfo.apiVersion = VK_API_VERSION_1_0;

       // Necessary for Vulkan Instances
       VkInstanceCreateInfo createInfo{}; 
       createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; 
       createInfo.pApplicationInfo = &appInfo;

       /*temp*/
       uint32_t glfwExtensionCount = 0;
       const char** glfwExtensions;

	   uint32_t extensionCount = 0; // Number of extensions supported by Vulkan
       vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr); // Retrieves the number of Vulkan supported extensions; first param is validation layer based filtering, and third is nullptr to get the count only
       std::vector<VkExtensionProperties> extensions(extensionCount);
	   vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()); // Get the extensions supported by Vulkan
       
       std::cout << "available extensions:\n";

       for (const auto& extension : extensions) {
           std::cout << '\t' << extension.extensionName << '\n';
       }
       /*end temp*/

       auto reqExtensions = getRequiredExtensions();
       createInfo.enabledExtensionCount = static_cast<uint32_t>(reqExtensions.size());
       createInfo.ppEnabledExtensionNames = reqExtensions.data();

       VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
       if (enableValidationLayers) {
           createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
           createInfo.ppEnabledLayerNames = validationLayers.data();

           populateDebugMessengerCreateInfo(debugCreateInfo);
           createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
       }
       else {
           createInfo.enabledLayerCount = 0;

           createInfo.pNext = nullptr;
       }

       if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
           throw std::runtime_error("failed to create instance!");
       }
   }

   SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
       SwapChainSupportDetails details;
       vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

       uint32_t formatCount;
       vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

       if (formatCount != 0) {
           details.formats.resize(formatCount);
           vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
       }

       uint32_t presentModeCount;
       vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

       if (presentModeCount != 0) {
           details.presentModes.resize(presentModeCount);
           vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
       }

       return details;
   }

   VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
       //VkSurfaceFormatKHR contains the format (color channel & types) and colorSpace member

       for (const auto& availableFormat : availableFormats) {
		   //if 8-bit BGRA is supported and SRGB color space is supported, return it
           if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
               return availableFormat;
           }
       }
	   //or just choose the first available format
       return availableFormats[0];

   }

   VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
       for (const auto& availablePresentMode : availablePresentModes) {
           if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
               return availablePresentMode;
           }
       }

       //guaranteed to be available: VSync
       return VK_PRESENT_MODE_FIFO_KHR;
   }

   VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
       //vulkan has already determined the pixel resolution of the window
       if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
           return capabilities.currentExtent;
       }
       else {
           //vulkan automatically sets the currentExtent.width/height to the maximum if the framebuffer size must be determined by actual physical window dimensions
           int width, height;
           glfwGetFramebufferSize(window, &width, &height);

           VkExtent2D actualExtent = {
               static_cast<uint32_t>(width),
               static_cast<uint32_t>(height)
           };

           //clamp bounds width to minimum and maximum extents
           actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width); 
           actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

           return actualExtent;
       }
   }


   bool checkValidationLayerSupport() {
       uint32_t layerCount;
       vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

       std::vector<VkLayerProperties> availableLayers(layerCount);
       vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()); 

       for (const char* validationLayer : validationLayers) {
           bool layerFound = false;
           for (const auto& availableLayer : availableLayers) {
               if (strcmp(validationLayer, availableLayer.layerName) == 0) { // Compare the layer names
                   layerFound = true; // If the layer is found, return true
                   break;
               }
           }

           if (!layerFound) {
               return false;
           }
       }

       return true;
   }

   std::vector<const char*> getRequiredExtensions() {
       uint32_t glfwExtensionCount = 0;
       const char** glfwExtensions;
       glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

       std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

       if (enableValidationLayers) {
           extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
       }

       return extensions;
   }

   static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
       VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
       VkDebugUtilsMessageTypeFlagsEXT messageType,
       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
       void* pUserData) {

       std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

       return VK_FALSE;
   }

    void mainLoop() {
		while (!glfwWindowShouldClose(window)) { // Check if the window should close
            glfwPollEvents(); // Poll for and process events
            drawFrame();
        }
        // Let the device finish operations and for it to idle before ending
        vkDeviceWaitIdle(device);
    }

};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}