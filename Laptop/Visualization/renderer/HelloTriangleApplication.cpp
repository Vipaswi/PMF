#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h> // GLFW library for window management and input handling

#include <iostream>  // reporting errors
#include <stdexcept> // propagating errors
#include <cstdlib>   // EXIT_SUCCESS, EXIT_FAILURE
#include <vector>
#include <optional> //for queue family indices: to determine queue nonexistence
#include <set>

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

    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; //implicitly destroyed btw
    VkDevice device; //a logical device that interfaces with the physical device
    VkQueue graphicsQueue; //handle to interface with graphics queue, implicitly cleaned up when the device is destroyed
    VkQueue presentQueue; 
	VkSurfaceKHR surface; //the surface to render to; requires the VK_KHR_win32_surface extension for Windows, added by glfw's required extensions

    void initWindow() {
		glfwInit(); // Initialize GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Specify that we don't want an OpenGL context
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Make the window not resizable for now.

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Width, height, title, monitor to open the window on, and OPENGL context to share with other windows (nullptr for no sharing).

    }

   void initVulkan() {
		createInstance(); // Create a Vulkan instance
		setupDebugMessenger(); // Setup the debug messenger for validation layers
        createSurface();
		pickPhysicalDevice();
		createLogicalDevice(); 
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

       return indices.isComplete() && extensionsSupported;
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

       return details;
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
        }
    }

    void cleanup() {
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr); // Destroy the Vulkan instance
        glfwDestroyWindow(window); //destroy the window

        glfwTerminate(); //terminate GLFW
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