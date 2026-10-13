#include "gpu_shader_vulkan.h"
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VULKAN_TRY(expression)                       \
    do                                               \
    {                                                \
        const VkResult vulkan_status = (expression); \
        if (VK_SUCCESS != vulkan_status)             \
        {                                            \
            status = vulkan_error(vulkan_status);    \
            goto cleanup;                            \
        }                                            \
    } while (0)

static error_t
vulkan_error(VkResult result)
{
    if (VK_ERROR_OUT_OF_HOST_MEMORY == result || VK_ERROR_OUT_OF_DEVICE_MEMORY == result)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    if (VK_ERROR_INCOMPATIBLE_DRIVER == result)
    {
        return POUND_ERROR_GPU_VULKAN_UNAVAILABLE;
    }
    return POUND_ERROR_GPU_VULKAN_FAILED;
}

static uint32_t
compute_family(VkPhysicalDevice device)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, NULL);
    if (0 == count || sizeof(VkQueueFamilyProperties) > SIZE_MAX / (size_t)count)
    {
        return UINT32_MAX;
    }
    VkQueueFamilyProperties *families = malloc((size_t)count * sizeof(*families));
    if (NULL == families)
    {
        return UINT32_MAX;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families);
    uint32_t result = UINT32_MAX;
    for (uint32_t index = 0; index < count; ++index)
    {
        if (0 != families[index].queueCount
            && 0 != (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            result = index;
            break;
        }
    }
    free(families);
    return result;
}

static error_t
select_device(VkInstance                  instance,
              uint32_t                    requested,
              uint32_t                    lane_count,
              size_t                      byte_size,
              VkPhysicalDevice           *selected,
              uint32_t                   *family,
              gpu_shader_vulkan_report_t *report)
{
    uint32_t count  = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (VK_SUCCESS != result)
    {
        return vulkan_error(result);
    }
    if (0 == count)
    {
        return POUND_ERROR_GPU_VULKAN_UNAVAILABLE;
    }
    if (requested != GPU_SHADER_VULKAN_AUTO_DEVICE && requested >= count)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (sizeof(VkPhysicalDevice) > SIZE_MAX / (size_t)count)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    VkPhysicalDevice *devices = malloc((size_t)count * sizeof(*devices));
    if (NULL == devices)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    result = vkEnumeratePhysicalDevices(instance, &count, devices);
    error_t status
        = VK_SUCCESS == result ? POUND_ERROR_GPU_VULKAN_UNAVAILABLE : vulkan_error(result);
    uint32_t best_score = 0;
    for (uint32_t index = 0; VK_SUCCESS == result && index < count; ++index)
    {
        if (requested != GPU_SHADER_VULKAN_AUTO_DEVICE && requested != index)
        {
            continue;
        }
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[index], &properties);
        if (properties.apiVersion < VK_API_VERSION_1_1
            || properties.limits.maxComputeWorkGroupCount[0] < lane_count
            || properties.limits.maxStorageBufferRange < byte_size)
        {
            continue;
        }
        const uint32_t queue = compute_family(devices[index]);
        const uint32_t score
            = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU == properties.deviceType ? 2U : 1U;
        if (queue == UINT32_MAX || score <= best_score)
        {
            continue;
        }
        best_score = score;
        *selected  = devices[index];
        *family    = queue;
        memcpy(report->device_name, properties.deviceName, sizeof(report->device_name));
        report->device_name[sizeof(report->device_name) - 1U] = '\0';
        report->device_index                                  = index;
        report->api_version                                   = properties.apiVersion;
        status                                                = POUND_SUCCESS;
    }
    free(devices);
    return status;
}

struct gpu_shader_vulkan_session
{
    VkInstance                 instance;
    VkDevice                   device;
    VkQueue                    queue;
    VkBuffer                   buffer;
    VkDeviceMemory             memory;
    VkDescriptorSetLayout      descriptor_layout;
    VkDescriptorPool           descriptor_pool;
    VkDescriptorSet            descriptor;
    VkPipelineLayout           pipeline_layout;
    VkCommandPool              command_pool;
    VkCommandBuffer            command;
    VkFence                    fence;
    void                      *mapped;
    gpu_shader_vulkan_report_t report;
    uint32_t                   lane_capacity;
    uint32_t                   kernel_count;
    uint32_t                   pending;
    uint32_t                   poisoned;
    uint64_t                   dispatches;
};

struct gpu_shader_vulkan_kernel
{
    gpu_shader_vulkan_session_t *session;
    VkPipeline                   pipeline;
    uint32_t                     budget;
};

static error_t
validate_state(uint32_t        budget,
               const uint32_t *state,
               size_t          state_word_count,
               uint32_t        lane_capacity,
               size_t         *lane_count)
{
    const uint32_t stride
        = budget == 0 ? GPU_SHADER_SPIRV_LANE_WORDS : GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS;
    if (NULL == state || state_word_count == 0 || state_word_count % stride != 0
        || budget > GPU_SHADER_SPIRV_BUDGET_LIMIT)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    *lane_count = state_word_count / stride;
    if (*lane_count > lane_capacity)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    if (budget == 0)
    {
        for (size_t index = 0; index < *lane_count; ++index)
        {
            if (state[index * stride + GPU_SHADER_ZERO_REGISTER] > 127U)
            {
                return POUND_ERROR_INVALID_ARGUMENT;
            }
        }
    }
    return POUND_SUCCESS;
}

static error_t
emit_words(const gpu_shader_program_t *program, uint32_t budget, uint32_t **words, size_t *count)
{
    size_t  word_count = 0;
    error_t status     = budget == 0
                             ? gpu_shader_spirv_emit(program, NULL, 0, &word_count)
                             : gpu_shader_spirv_emit_bounded(program, budget, NULL, 0, &word_count);
    if (status != POUND_SUCCESS)
    {
        return status;
    }
    if (word_count > SIZE_MAX / sizeof(uint32_t))
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    uint32_t *storage = malloc(word_count * sizeof(*storage));
    if (NULL == storage)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    status = budget == 0
                 ? gpu_shader_spirv_emit(program, storage, word_count, &word_count)
                 : gpu_shader_spirv_emit_bounded(program, budget, storage, word_count, &word_count);
    if (status != POUND_SUCCESS)
    {
        free(storage);
        return status;
    }
    *words = storage;
    *count = word_count;
    return POUND_SUCCESS;
}

static void
drain_pending(gpu_shader_vulkan_session_t *session)
{
    while (session->pending != 0)
    {
        VkResult result = vkDeviceWaitIdle(session->device);
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST)
        {
            result
                = vkWaitForFences(session->device, 1, &session->fence, VK_TRUE, UINT64_C(10000000));
        }
        if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST)
        {
            session->pending = 0;
        }
    }
}

static void
destroy_resources(gpu_shader_vulkan_session_t *session)
{
    if (session->device != VK_NULL_HANDLE)
    {
        drain_pending(session);
        if (session->fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(session->device, session->fence, NULL);
        }
        if (session->command_pool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(session->device, session->command_pool, NULL);
        }
        if (session->descriptor_pool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(session->device, session->descriptor_pool, NULL);
        }
        if (session->pipeline_layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(session->device, session->pipeline_layout, NULL);
        }
        if (session->descriptor_layout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(session->device, session->descriptor_layout, NULL);
        }
        if (session->mapped != NULL)
        {
            vkUnmapMemory(session->device, session->memory);
        }
        if (session->buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(session->device, session->buffer, NULL);
        }
        if (session->memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(session->device, session->memory, NULL);
        }
        vkDestroyDevice(session->device, NULL);
    }
    if (session->instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(session->instance, NULL);
    }
    free(session);
}

error_t
gpu_shader_vulkan_session_create(uint32_t                      device_index,
                                 uint32_t                      lane_capacity,
                                 gpu_shader_vulkan_session_t **output)
{
    if (NULL == output || lane_capacity == 0)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (lane_capacity > GPU_SHADER_VULKAN_LANE_LIMIT)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    gpu_shader_vulkan_session_t *session = calloc(1, sizeof(*session));
    if (NULL == session)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    error_t status         = POUND_SUCCESS;
    session->lane_capacity = lane_capacity;
    const size_t byte_size
        = (size_t)lane_capacity * GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS * sizeof(uint32_t);
    const VkApplicationInfo    application = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                               .pApplicationName = "Pound integer compute prototype",
                                               .apiVersion = VK_API_VERSION_1_1 };
    const VkInstanceCreateInfo instance_info
        = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &application };
    VULKAN_TRY(vkCreateInstance(&instance_info, NULL, &session->instance));
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t         family   = UINT32_MAX;
    status                    = select_device(session->instance,
                           device_index,
                           lane_capacity,
                           byte_size,
                           &physical,
                           &family,
                           &session->report);
    if (status != POUND_SUCCESS)
    {
        goto cleanup;
    }
    const float                   priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info
        = { .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount       = 1,
            .pQueuePriorities = &priority };
    const VkDeviceCreateInfo device_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                             .queueCreateInfoCount = 1,
                                             .pQueueCreateInfos    = &queue_info };
    VULKAN_TRY(vkCreateDevice(physical, &device_info, NULL, &session->device));
    vkGetDeviceQueue(session->device, family, 0, &session->queue);
    const VkBufferCreateInfo buffer_info = { .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                             .size        = byte_size,
                                             .usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VULKAN_TRY(vkCreateBuffer(session->device, &buffer_info, NULL, &session->buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(session->device, session->buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    uint32_t memory_type = UINT32_MAX;
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index)
    {
        const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[index].propertyFlags;
        if ((requirements.memoryTypeBits & (1U << index)) != 0
            && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
        {
            memory_type = index;
            if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
            {
                break;
            }
        }
    }
    if (memory_type == UINT32_MAX)
    {
        status = POUND_ERROR_GPU_VULKAN_UNAVAILABLE;
        goto cleanup;
    }
    session->report.coherent_memory = (memory_properties.memoryTypes[memory_type].propertyFlags
                                       & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                      != 0;
    const VkMemoryAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                              .allocationSize  = requirements.size,
                                              .memoryTypeIndex = memory_type };
    VULKAN_TRY(vkAllocateMemory(session->device, &allocation, NULL, &session->memory));
    VULKAN_TRY(vkBindBufferMemory(session->device, session->buffer, session->memory, 0));
    VULKAN_TRY(
        vkMapMemory(session->device, session->memory, 0, VK_WHOLE_SIZE, 0, &session->mapped));
    const VkDescriptorSetLayoutBinding binding
        = { .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT };
    const VkDescriptorSetLayoutCreateInfo layout_info
        = { .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings    = &binding };
    VULKAN_TRY(vkCreateDescriptorSetLayout(
        session->device, &layout_info, NULL, &session->descriptor_layout));
    const VkPipelineLayoutCreateInfo pipeline_layout_info
        = { .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts    = &session->descriptor_layout };
    VULKAN_TRY(vkCreatePipelineLayout(
        session->device, &pipeline_layout_info, NULL, &session->pipeline_layout));
    const VkDescriptorPoolSize pool_size
        = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
    const VkDescriptorPoolCreateInfo pool_info
        = { .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets       = 1,
            .poolSizeCount = 1,
            .pPoolSizes    = &pool_size };
    VULKAN_TRY(
        vkCreateDescriptorPool(session->device, &pool_info, NULL, &session->descriptor_pool));
    const VkDescriptorSetAllocateInfo descriptor_info
        = { .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = session->descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &session->descriptor_layout };
    VULKAN_TRY(vkAllocateDescriptorSets(session->device, &descriptor_info, &session->descriptor));
    const VkDescriptorBufferInfo descriptor_buffer
        = { .buffer = session->buffer, .range = byte_size };
    const VkWriteDescriptorSet descriptor_write
        = { .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = session->descriptor,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &descriptor_buffer };
    vkUpdateDescriptorSets(session->device, 1, &descriptor_write, 0, NULL);
    const VkCommandPoolCreateInfo command_pool_info
        = { .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = family };
    VULKAN_TRY(
        vkCreateCommandPool(session->device, &command_pool_info, NULL, &session->command_pool));
    const VkCommandBufferAllocateInfo command_info
        = { .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = session->command_pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1 };
    VULKAN_TRY(vkAllocateCommandBuffers(session->device, &command_info, &session->command));
    const VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VULKAN_TRY(vkCreateFence(session->device, &fence_info, NULL, &session->fence));
    *output = session;
    return POUND_SUCCESS;

cleanup:
    destroy_resources(session);
    return status;
}

error_t
gpu_shader_vulkan_session_destroy(gpu_shader_vulkan_session_t *session)
{
    if (NULL == session)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (session->kernel_count != 0)
    {
        return POUND_ERROR_GPU_BUSY;
    }
    destroy_resources(session);
    return POUND_SUCCESS;
}

static error_t
compile_kernel(gpu_shader_vulkan_session_t *session,
               const uint32_t              *words,
               size_t                       count,
               uint32_t                     budget,
               gpu_shader_vulkan_kernel_t **output)
{
    gpu_shader_vulkan_kernel_t *kernel = calloc(1, sizeof(*kernel));
    if (NULL == kernel)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    kernel->session                       = session;
    kernel->budget                        = budget;
    VkShaderModule                 module = VK_NULL_HANDLE;
    error_t                        status = POUND_SUCCESS;
    const VkShaderModuleCreateInfo module_info
        = { .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = count * sizeof(*words),
            .pCode    = words };
    VULKAN_TRY(vkCreateShaderModule(session->device, &module_info, NULL, &module));
    const VkComputePipelineCreateInfo pipeline_info
        = { .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = { .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                        .module = module,
                        .pName  = "main" },
            .layout = session->pipeline_layout };
    VULKAN_TRY(vkCreateComputePipelines(
        session->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &kernel->pipeline));
    vkDestroyShaderModule(session->device, module, NULL);
    ++session->kernel_count;
    *output = kernel;
    return POUND_SUCCESS;

cleanup:
    session->poisoned = 1;
    if (kernel->pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(session->device, kernel->pipeline, NULL);
    }
    if (module != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(session->device, module, NULL);
    }
    free(kernel);
    return status;
}

error_t
gpu_shader_vulkan_kernel_create(gpu_shader_vulkan_session_t *session,
                                const gpu_shader_program_t  *program,
                                uint32_t                     budget,
                                gpu_shader_vulkan_kernel_t **output)
{
    if (NULL == session || NULL == output || budget > GPU_SHADER_SPIRV_BUDGET_LIMIT)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (session->poisoned != 0)
    {
        return POUND_ERROR_GPU_VULKAN_FAILED;
    }
    if (session->kernel_count == UINT32_MAX)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    uint32_t *words  = NULL;
    size_t    count  = 0;
    error_t   status = emit_words(program, budget, &words, &count);
    if (status == POUND_SUCCESS)
    {
        status = compile_kernel(session, words, count, budget, output);
    }
    free(words);
    return status;
}

void
gpu_shader_vulkan_kernel_destroy(gpu_shader_vulkan_kernel_t *kernel)
{
    if (NULL == kernel)
    {
        return;
    }
    gpu_shader_vulkan_session_t *session = kernel->session;
    drain_pending(session);
    vkDestroyPipeline(session->device, kernel->pipeline, NULL);
    --session->kernel_count;
    free(kernel);
}

error_t
gpu_shader_vulkan_dispatch(gpu_shader_vulkan_session_t      *session,
                           const gpu_shader_vulkan_kernel_t *kernel,
                           uint32_t                         *state,
                           size_t                            state_word_count,
                           gpu_shader_vulkan_report_t       *report)
{
    if (NULL == session || NULL == kernel || kernel->session != session)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    size_t  lane_count = 0;
    error_t status     = validate_state(
        kernel->budget, state, state_word_count, session->lane_capacity, &lane_count);
    if (status != POUND_SUCCESS)
    {
        return status;
    }
    if (session->dispatches == UINT64_MAX)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    if (session->poisoned != 0)
    {
        return POUND_ERROR_GPU_VULKAN_FAILED;
    }
    const size_t byte_size = state_word_count * sizeof(*state);
    memcpy(session->mapped, state, byte_size);
    const VkMappedMemoryRange mapped_range = { .sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                               .memory = session->memory,
                                               .size   = VK_WHOLE_SIZE };
    if (session->report.coherent_memory == 0)
    {
        VULKAN_TRY(vkFlushMappedMemoryRanges(session->device, 1, &mapped_range));
    }
    VULKAN_TRY(vkResetCommandBuffer(session->command, 0));
    VULKAN_TRY(vkResetFences(session->device, 1, &session->fence));
    const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                             .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VULKAN_TRY(vkBeginCommandBuffer(session->command, &begin));
    const VkMemoryBarrier upload
        = { .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(session->command,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1,
                         &upload,
                         0,
                         NULL,
                         0,
                         NULL);
    vkCmdBindPipeline(session->command, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(session->command,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            session->pipeline_layout,
                            0,
                            1,
                            &session->descriptor,
                            0,
                            NULL);
    vkCmdDispatch(session->command, (uint32_t)lane_count, 1, 1);
    const VkMemoryBarrier download = { .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                       .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                       .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(session->command,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         1,
                         &download,
                         0,
                         NULL,
                         0,
                         NULL);
    VULKAN_TRY(vkEndCommandBuffer(session->command));
    const VkSubmitInfo submit = { .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers    = &session->command };
    session->pending          = 1;
    VULKAN_TRY(vkQueueSubmit(session->queue, 1, &submit, session->fence));
    VULKAN_TRY(vkWaitForFences(session->device, 1, &session->fence, VK_TRUE, UINT64_MAX));
    session->pending = 0;
    if (session->report.coherent_memory == 0)
    {
        VULKAN_TRY(vkInvalidateMappedMemoryRanges(session->device, 1, &mapped_range));
    }
    ++session->dispatches;
    memcpy(state, session->mapped, byte_size);
    if (NULL != report)
    {
        *report                = session->report;
        report->lane_count     = (uint32_t)lane_count;
        report->lane_words     = kernel->budget == 0 ? GPU_SHADER_SPIRV_LANE_WORDS
                                                     : GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS;
        report->dispatch_index = session->dispatches;
    }
    return POUND_SUCCESS;

cleanup:
    session->poisoned = 1;
    return status;
}

error_t
gpu_shader_vulkan_run(const gpu_shader_program_t *program,
                      uint32_t                    budget,
                      uint32_t                   *state,
                      size_t                      state_word_count,
                      uint32_t                    device_index,
                      gpu_shader_vulkan_report_t *report)
{
    size_t  lane_count = 0;
    error_t status     = validate_state(
        budget, state, state_word_count, GPU_SHADER_VULKAN_LANE_LIMIT, &lane_count);
    if (status != POUND_SUCCESS)
    {
        return status;
    }
    uint32_t *words = NULL;
    size_t    count = 0;
    status          = emit_words(program, budget, &words, &count);
    if (status != POUND_SUCCESS)
    {
        return status;
    }
    gpu_shader_vulkan_session_t *session = NULL;
    gpu_shader_vulkan_kernel_t  *kernel  = NULL;
    status = gpu_shader_vulkan_session_create(device_index, (uint32_t)lane_count, &session);
    if (status == POUND_SUCCESS)
    {
        status = compile_kernel(session, words, count, budget, &kernel);
        if (status == POUND_SUCCESS)
        {
            status = gpu_shader_vulkan_dispatch(session, kernel, state, state_word_count, report);
        }
        gpu_shader_vulkan_kernel_destroy(kernel);
        gpu_shader_vulkan_session_destroy(session);
    }
    free(words);
    return status;
}
