#ifndef POUND_GPU_SHADER_SPIRV_H
#define POUND_GPU_SHADER_SPIRV_H

#include "gpu_shader.h"

#define GPU_SHADER_SPIRV_INSTRUCTION_LIMIT  1024U
#define GPU_SHADER_SPIRV_LANE_WORDS         256U
#define GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS 261U
#define GPU_SHADER_SPIRV_BUDGET_LIMIT       4096U

typedef enum
{
    GPU_SHADER_SPIRV_PC = 256,
    GPU_SHADER_SPIRV_STEPS_LOW,
    GPU_SHADER_SPIRV_STEPS_HIGH,
    GPU_SHADER_SPIRV_HALTED,
    GPU_SHADER_SPIRV_STATUS,
} gpu_shader_spirv_state_word_t;

error_t gpu_shader_spirv_emit(const gpu_shader_program_t *program,
                              uint32_t                   *words,
                              size_t                      capacity,
                              size_t                     *word_count);
error_t gpu_shader_spirv_emit_bounded(const gpu_shader_program_t *program,
                                      uint32_t                    budget,
                                      uint32_t                   *words,
                                      size_t                      capacity,
                                      size_t                     *word_count);

#endif
