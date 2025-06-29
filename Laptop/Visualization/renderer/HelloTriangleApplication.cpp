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

        // Defines the input-rate and size of vertex data
        static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription bindingDescription{};
            bindingDescription.binding = 0; // Binding index in binding arr.
            bindingDescription.stride = sizeof(Vertex); // # bytes bet entries
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;  // When to move data; current: after each vertex

            return bindingDescription;
        };

        // How to extract vertex attributes from chunks of vertex data from binding descriptions
        static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
            std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
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
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}},
        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
    };
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    // Index Buffer:
    const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;


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
		createGraphicsPipeline(); 
		createFramebuffers(); 
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createCommandBuffers();
        createSyncObjects();
    }

    void cleanup() {
        cleanupSwapChain();

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
        // Allocate a temporary command buffer to submit the transfer operation
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        // Allocate buffer:
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        // Begin the command, telling it that it's only going to run once
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo); // Start recording to the command buffer

        
        // Specify the region to copy, and set the command to copy the buffer
            // -> Note: can't use VK_WHOLE_SIZE here
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        vkEndCommandBuffer(commandBuffer); // Stop recording to the command buffer

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        // Submit command to graphics queue
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue); // Alt. for multiple transfers use fences :thumbsup:
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

    void drawFrame() {
        // Wait for previous frame to finish
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

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
	   rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; //clockwise winding order for front faces; counter-clockwise is default
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
       pipelineLayoutInfo.setLayoutCount = 0; // Optional
       pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
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

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size()); 
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            //how data should be interpreted
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; //can be 1d,2d, or 3d
            createInfo.format = swapChainImageFormat;
            //default mapping; we can also map them from 0 to 1
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            //Image purpose and image part accessed
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
			//create the image view for the swapchain image
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
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

       return indices.isComplete() && extensionsSupported && swapChainAdequate;
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