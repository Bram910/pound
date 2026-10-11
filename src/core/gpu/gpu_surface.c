#include "gpu_surface.h"

error_t
gpu_surface_prepare(const gpu_surface_t  *surface,
                    uint32_t              width,
                    uint32_t              rows,
                    gpu_surface_layout_t *layout)
{
    if (NULL == surface || NULL == layout)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const uint32_t height_log2 = (surface->block_size >> 4U) & 15U;
    const uint32_t depth_log2  = (surface->block_size >> 8U) & 15U;
    if (0x1000U != (surface->block_size & ~0xFF0U) || height_log2 > 5U || depth_log2 > 5U)
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    if (0 == surface->width || 0 == surface->height || 0 == surface->depth
        || surface->layer >= surface->depth || surface->origin_x >= surface->width
        || surface->origin_y >= surface->height || width > surface->width - surface->origin_x
        || rows > surface->height - surface->origin_y)
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    const uint32_t block_height = 8U << height_log2;
    const uint32_t block_depth  = 1U << depth_log2;
    const uint64_t slice_stride = UINT64_C(512) << height_log2;
    const uint64_t block_stride = slice_stride * block_depth;
    const uint64_t row_stride   = (((uint64_t)surface->width + 63U) / 64U) * block_stride;
    const uint64_t block_rows   = ((uint64_t)surface->height + block_height - 1U) / block_height;
    if (block_rows > GPU_MEMORY_ADDRESS_LIMIT / row_stride)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    const uint64_t plane_stride = block_rows * row_stride;
    const uint64_t depth_groups = ((uint64_t)surface->depth + block_depth - 1U) / block_depth;
    if (depth_groups > GPU_MEMORY_ADDRESS_LIMIT / plane_stride)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    const uint64_t slice_offset = (surface->layer / block_depth) * plane_stride
                                  + (surface->layer % block_depth) * slice_stride;
    *layout = (gpu_surface_layout_t) { .block_stride  = block_stride,
                                       .row_stride    = row_stride,
                                       .slice_offset  = slice_offset,
                                       .block_height  = block_height,
                                       .origin_x      = surface->origin_x,
                                       .origin_y      = surface->origin_y,
                                       .sector_layout = GPU_SECTOR_LAYOUT_16BX2 };
    return POUND_SUCCESS;
}

error_t
gpu_surface_prepare_elements(const gpu_surface_t  *surface,
                             uint32_t              width,
                             uint32_t              rows,
                             uint32_t              element_size,
                             gpu_surface_layout_t *layout)
{
    if (NULL == surface || NULL == layout || 0 == element_size || element_size > 16U)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (surface->width > UINT32_MAX / element_size || surface->origin_x > UINT32_MAX / element_size
        || width > UINT32_MAX / element_size)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    gpu_surface_t bytes = *surface;
    bytes.width *= element_size;
    bytes.origin_x *= element_size;
    return gpu_surface_prepare(&bytes, width * element_size, rows, layout);
}

uint64_t
gpu_surface_offset(const gpu_surface_layout_t *layout, uint32_t column, uint32_t row)
{
    const uint64_t horizontal = (uint64_t)layout->origin_x + column;
    const uint64_t vertical   = (uint64_t)layout->origin_y + row;
    const uint64_t gob        = layout->slice_offset
                         + (vertical / layout->block_height) * layout->row_stride
                         + (horizontal / 64U) * layout->block_stride
                         + ((vertical % layout->block_height) / 8U) * 512U;
    const uint64_t sector
        = GPU_SECTOR_LAYOUT_TURING2D == layout->sector_layout
              ? ((horizontal & 32U) << 3U) | ((vertical & 4U) << 5U) | ((horizontal & 16U) << 2U)
                    | ((vertical & 3U) << 4U) | (horizontal & 15U)
              : ((horizontal & 32U) << 3U) | ((vertical & 6U) << 5U) | ((horizontal & 16U) << 1U)
                    | ((vertical & 1U) << 4U) | (horizontal & 15U);
    return gob + sector;
}
