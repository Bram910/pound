#ifndef POUND_GPU_SURFACE_H
#define POUND_GPU_SURFACE_H

#include "gpu_memory.h"

typedef enum
{
    GPU_SECTOR_LAYOUT_UNSPECIFIED,
    GPU_SECTOR_LAYOUT_16BX2,
    GPU_SECTOR_LAYOUT_TURING2D
} gpu_sector_layout_t;

typedef struct
{
    uint32_t block_size;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t layer;
    uint32_t origin_x;
    uint32_t origin_y;
} gpu_surface_t;

typedef struct
{
    uint64_t            block_stride;
    uint64_t            row_stride;
    uint64_t            slice_offset;
    uint32_t            block_height;
    uint32_t            origin_x;
    uint32_t            origin_y;
    gpu_sector_layout_t sector_layout;
} gpu_surface_layout_t;

error_t  gpu_surface_prepare(const gpu_surface_t  *surface,
                             uint32_t              width,
                             uint32_t              rows,
                             gpu_surface_layout_t *layout);
uint64_t gpu_surface_offset(const gpu_surface_layout_t *layout, uint32_t column, uint32_t row);
error_t  gpu_surface_prepare_elements(const gpu_surface_t  *surface,
                                      uint32_t              width,
                                      uint32_t              rows,
                                      uint32_t              element_size,
                                      gpu_surface_layout_t *layout);

#endif
