#ifndef POUND_GPU_SHADER_SPIRV_H
#define POUND_GPU_SHADER_SPIRV_H

#include "gpu_shader.h"

#define GPU_SHADER_SPIRV_INSTRUCTION_LIMIT 1024U
#define GPU_SHADER_SPIRV_LANE_WORDS        256U

error_t gpu_shader_spirv_emit(const gpu_shader_program_t *program,
                              uint32_t                   *words,
                              size_t                      capacity,
                              size_t                     *word_count);

#endif
