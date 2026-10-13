#ifndef POUND_GPU_SHADER_VULKAN_H
#define POUND_GPU_SHADER_VULKAN_H

#include "gpu_shader_spirv.h"

#define GPU_SHADER_VULKAN_AUTO_DEVICE UINT32_MAX
#define GPU_SHADER_VULKAN_LANE_LIMIT  4096U

typedef struct gpu_shader_vulkan_session gpu_shader_vulkan_session_t;
typedef struct gpu_shader_vulkan_kernel  gpu_shader_vulkan_kernel_t;

typedef struct
{
    char     device_name[256];
    uint32_t device_index;
    uint32_t api_version;
    uint32_t coherent_memory;
    uint32_t lane_count;
    uint32_t lane_words;
    uint64_t dispatch_index;
} gpu_shader_vulkan_report_t;

error_t gpu_shader_vulkan_session_create(uint32_t                      device_index,
                                         uint32_t                      lane_capacity,
                                         gpu_shader_vulkan_session_t **session);
error_t gpu_shader_vulkan_session_destroy(gpu_shader_vulkan_session_t *session);
error_t gpu_shader_vulkan_kernel_create(gpu_shader_vulkan_session_t *session,
                                        const gpu_shader_program_t  *program,
                                        uint32_t                     budget,
                                        gpu_shader_vulkan_kernel_t **kernel);
void    gpu_shader_vulkan_kernel_destroy(gpu_shader_vulkan_kernel_t *kernel);
error_t gpu_shader_vulkan_dispatch(gpu_shader_vulkan_session_t      *session,
                                   const gpu_shader_vulkan_kernel_t *kernel,
                                   uint32_t                         *state,
                                   size_t                            state_word_count,
                                   gpu_shader_vulkan_report_t       *report);

error_t gpu_shader_vulkan_run(const gpu_shader_program_t *program,
                              uint32_t                    budget,
                              uint32_t                   *state,
                              size_t                      state_word_count,
                              uint32_t                    device_index,
                              gpu_shader_vulkan_report_t *report);

#endif
