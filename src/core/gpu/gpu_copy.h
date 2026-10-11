#ifndef POUND_GPU_COPY_H
#define POUND_GPU_COPY_H

#include "gpu_surface.h"

#define GPU_CLASS_AMPERE_DMA_COPY 0xC6B5U

typedef struct
{
    uint64_t            source;
    uint64_t            destination;
    uint64_t            semaphore_address;
    uint32_t            semaphore_payload;
    uint32_t            source_pitch;
    uint32_t            destination_pitch;
    uint32_t            line_length;
    uint32_t            line_count;
    gpu_sector_layout_t sector_layout;
    gpu_surface_t       source_surface;
    gpu_surface_t       destination_surface;
    uint32_t            remap_constant_a;
    uint32_t            remap_constant_b;
    uint32_t            remap_components;
} gpu_copy_t;

error_t gpu_copy_method(gpu_copy_t         *copy,
                        const gpu_memory_t *memory,
                        uint32_t            method,
                        uint32_t            data,
                        void               *scratch,
                        size_t              scratch_size);

#endif
