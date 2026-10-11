#ifndef POUND_GPU_INLINE_H
#define POUND_GPU_INLINE_H

#include "gpu_surface.h"

#define GPU_CLASS_KEPLER_INLINE_TO_MEMORY 0xA140U

typedef struct
{
    uint64_t             destination;
    uint64_t             semaphore_address;
    uint64_t             bytes_written;
    uint64_t             total_bytes;
    uint32_t             semaphore_payload;
    uint32_t             pitch;
    uint32_t             line_length;
    uint32_t             line_count;
    uint32_t             launch_flags;
    uint32_t             active;
    gpu_sector_layout_t  sector_layout;
    gpu_surface_t        surface;
    gpu_surface_layout_t layout;
} gpu_inline_t;

error_t gpu_inline_method(gpu_inline_t       *upload,
                          const gpu_memory_t *memory,
                          uint32_t            method,
                          uint32_t            data,
                          size_t              transfer_capacity);

#endif
