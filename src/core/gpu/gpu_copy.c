#include "gpu_copy.h"

static error_t
set_upper(uint64_t *address, uint32_t upper)
{
    if (upper > 0x1FFFFU)
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    *address = (*address & UINT64_C(0xFFFFFFFF)) | ((uint64_t)upper << 32U);
    return POUND_SUCCESS;
}

static error_t
access_rows(const gpu_memory_t         *memory,
            uint64_t                    address,
            uint32_t                    pitch,
            const gpu_surface_layout_t *layout,
            uint32_t                    width,
            uint32_t                    rows,
            uint8_t                    *buffer,
            uint32_t                    action)
{
    if (address >= GPU_MEMORY_ADDRESS_LIMIT)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    uint8_t *cursor = buffer;
    for (uint32_t row = 0; row < rows; ++row)
    {
        uint32_t column = 0;
        while (column < width)
        {
            const uint64_t offset = NULL == layout ? (uint64_t)row * pitch + column
                                                   : gpu_surface_offset(layout, column, row);
            if (offset >= GPU_MEMORY_ADDRESS_LIMIT - address)
            {
                return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
            }
            uint32_t length = width - column;
            if (NULL != layout)
            {
                const uint32_t sector_remaining = 16U - ((layout->origin_x + column) & 15U);
                if (length > sector_remaining)
                {
                    length = sector_remaining;
                }
            }
            error_t status;
            if (GPU_MEMORY_READ == action)
            {
                status = gpu_memory_read(memory, address + offset, cursor, length);
            }
            else if (GPU_MEMORY_WRITE == action)
            {
                status = gpu_memory_write(memory, address + offset, cursor, length);
            }
            else
            {
                status = gpu_memory_validate(memory, address + offset, length, GPU_MEMORY_WRITE);
            }
            if (POUND_SUCCESS != status)
            {
                return status;
            }
            column += length;
            cursor += length;
        }
    }
    return POUND_SUCCESS;
}

static error_t
access_remapped_destination(const gpu_copy_t           *copy,
                            const gpu_memory_t         *memory,
                            const gpu_surface_layout_t *layout,
                            uint32_t                    rows,
                            const uint8_t              *input,
                            uint32_t                    source_stride,
                            uint32_t                    destination_stride,
                            uint32_t                    component_size,
                            uint32_t                    destination_components,
                            uint32_t                    write)
{
    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint8_t *cursor = input;
        if (NULL != input)
        {
            cursor += (size_t)row * copy->line_length * source_stride;
        }
        for (uint32_t pixel = 0; pixel < copy->line_length; ++pixel)
        {
            for (uint32_t component = 0; component < destination_components; ++component)
            {
                const uint32_t selector = (copy->remap_components >> (component * 4U)) & 7U;
                if (6U == selector)
                {
                    continue;
                }
                uint8_t        constant[4];
                const uint8_t *value = NULL;
                if (0 != write)
                {
                    if (selector < 4U)
                    {
                        value = cursor + selector * component_size;
                    }
                    else
                    {
                        const uint32_t payload
                            = 4U == selector ? copy->remap_constant_a : copy->remap_constant_b;
                        for (uint32_t index = 0; index < component_size; ++index)
                        {
                            constant[index] = (uint8_t)(payload >> (index * 8U));
                        }
                        value = constant;
                    }
                }
                const uint32_t column   = pixel * destination_stride + component * component_size;
                uint32_t       consumed = 0;
                while (consumed < component_size)
                {
                    const uint64_t offset
                        = NULL == layout
                              ? (uint64_t)row * copy->destination_pitch + column + consumed
                              : gpu_surface_offset(layout, column + consumed, row);
                    if (offset >= GPU_MEMORY_ADDRESS_LIMIT - copy->destination)
                    {
                        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
                    }
                    uint32_t length = component_size - consumed;
                    if (NULL != layout)
                    {
                        const uint32_t available
                            = 16U - ((layout->origin_x + column + consumed) & 15U);
                        if (length > available)
                        {
                            length = available;
                        }
                    }
                    const uint64_t address = copy->destination + offset;
                    const error_t  status
                        = 0 == write
                              ? gpu_memory_validate(memory, address, length, GPU_MEMORY_WRITE)
                              : gpu_memory_write(memory, address, value + consumed, length);
                    if (POUND_SUCCESS != status)
                    {
                        return status;
                    }
                    consumed += length;
                }
            }
            if (NULL != cursor)
            {
                cursor += source_stride;
            }
        }
    }
    return POUND_SUCCESS;
}

static error_t
remap_copy(const gpu_copy_t   *copy,
           const gpu_memory_t *memory,
           uint32_t            launch,
           uint32_t            rows,
           void               *scratch,
           size_t              scratch_size)
{
    if (0 != (copy->remap_components & ~0x03337777U))
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    const uint32_t component_size         = ((copy->remap_components >> 16U) & 3U) + 1U;
    const uint32_t source_components      = ((copy->remap_components >> 20U) & 3U) + 1U;
    const uint32_t destination_components = ((copy->remap_components >> 24U) & 3U) + 1U;
    const uint32_t source_stride          = source_components * component_size;
    const uint32_t destination_stride     = destination_components * component_size;
    uint32_t       reads_source           = 0;
    uint32_t       writes_destination     = 0;
    for (uint32_t component = 0; component < destination_components; ++component)
    {
        const uint32_t selector = (copy->remap_components >> (component * 4U)) & 7U;
        if (7U == selector || (selector < 4U && selector >= source_components))
        {
            return POUND_ERROR_GPU_INVALID_COMMAND;
        }
        reads_source |= selector < 4U;
        writes_destination |= selector < 6U;
    }
    if (0 == writes_destination)
    {
        return POUND_SUCCESS;
    }
    const uint32_t source_tiled      = 0 != reads_source && 0 == (launch & 0x80U);
    const uint32_t destination_tiled = 0 == (launch & 0x100U);
    if ((0 != source_tiled || 0 != destination_tiled)
        && ((GPU_SECTOR_LAYOUT_16BX2 != copy->sector_layout
             && GPU_SECTOR_LAYOUT_TURING2D != copy->sector_layout)
            || 0 == (launch & 0x200U)))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    const uint64_t source_width      = (uint64_t)copy->line_length * source_stride;
    const uint64_t destination_width = (uint64_t)copy->line_length * destination_stride;
    if (source_width > UINT32_MAX || destination_width > UINT32_MAX
        || destination_width > scratch_size / rows
        || (0 != reads_source && source_width > scratch_size / rows))
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    if (copy->destination >= GPU_MEMORY_ADDRESS_LIMIT)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    gpu_surface_layout_t        source_layout;
    gpu_surface_layout_t        destination_layout;
    const gpu_surface_layout_t *source      = NULL;
    const gpu_surface_layout_t *destination = NULL;
    if (0 != source_tiled)
    {
        const error_t status = gpu_surface_prepare_elements(
            &copy->source_surface, copy->line_length, rows, source_stride, &source_layout);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        source_layout.sector_layout = copy->sector_layout;
        source                      = &source_layout;
    }
    if (0 != destination_tiled)
    {
        const error_t status = gpu_surface_prepare_elements(&copy->destination_surface,
                                                            copy->line_length,
                                                            rows,
                                                            destination_stride,
                                                            &destination_layout);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        destination_layout.sector_layout = copy->sector_layout;
        destination                      = &destination_layout;
    }
    else
    {
        const uint64_t last_row = (uint64_t)(rows - 1U) * copy->destination_pitch;
        if (last_row >= GPU_MEMORY_ADDRESS_LIMIT - copy->destination
            || destination_width > GPU_MEMORY_ADDRESS_LIMIT - copy->destination - last_row)
        {
            return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
        }
    }
    const uint8_t *input = NULL;
    if (0 != reads_source)
    {
        if (NULL == scratch)
        {
            return POUND_ERROR_INVALID_ARGUMENT;
        }
        const error_t status = access_rows(memory,
                                           copy->source,
                                           copy->source_pitch,
                                           source,
                                           (uint32_t)source_width,
                                           rows,
                                           scratch,
                                           GPU_MEMORY_READ);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        input = scratch;
    }
    error_t status = access_remapped_destination(copy,
                                                 memory,
                                                 destination,
                                                 rows,
                                                 input,
                                                 source_stride,
                                                 destination_stride,
                                                 component_size,
                                                 destination_components,
                                                 0);
    if (POUND_SUCCESS == status)
    {
        status = access_remapped_destination(copy,
                                             memory,
                                             destination,
                                             rows,
                                             input,
                                             source_stride,
                                             destination_stride,
                                             component_size,
                                             destination_components,
                                             1U);
    }
    return status;
}

static error_t
launch_copy(const gpu_copy_t   *copy,
            const gpu_memory_t *memory,
            uint32_t            launch,
            void               *scratch,
            size_t              scratch_size)
{
    const uint32_t transfer  = launch & 3U;
    const uint32_t semaphore = (launch >> 3U) & 3U;
    if (3U == transfer)
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    if (0 != (launch & ~0x79FU) || semaphore > 1U)
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    if (0 != semaphore)
    {
        if (0 != (copy->semaphore_address & 3U))
        {
            return POUND_ERROR_GPU_INVALID_COMMAND;
        }
        const error_t status
            = gpu_memory_validate(memory, copy->semaphore_address, 4U, GPU_MEMORY_WRITE);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
    }

    const uint32_t rows  = 0 != (launch & 0x200U) ? copy->line_count : 1U;
    const uint32_t width = copy->line_length;
    if (0 != transfer && 0 != width && 0 != rows && 0 != (launch & 0x400U))
    {
        const error_t status = remap_copy(copy, memory, launch, rows, scratch, scratch_size);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
    }
    else if (0 != transfer && 0 != width && 0 != rows)
    {
        if (width > scratch_size / rows)
        {
            return POUND_ERROR_GPU_RESOURCE_LIMIT;
        }
        if (NULL == scratch)
        {
            return POUND_ERROR_INVALID_ARGUMENT;
        }
        gpu_surface_layout_t        source_layout;
        gpu_surface_layout_t        destination_layout;
        const gpu_surface_layout_t *source      = NULL;
        const gpu_surface_layout_t *destination = NULL;
        if (0x180U != (launch & 0x180U))
        {
            if ((GPU_SECTOR_LAYOUT_16BX2 != copy->sector_layout
                 && GPU_SECTOR_LAYOUT_TURING2D != copy->sector_layout)
                || 0 == (launch & 0x200U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
            }
            if (0 == (launch & 0x80U))
            {
                const error_t status
                    = gpu_surface_prepare(&copy->source_surface, width, rows, &source_layout);
                if (POUND_SUCCESS != status)
                {
                    return status;
                }
                source_layout.sector_layout = copy->sector_layout;
                source                      = &source_layout;
            }
            if (0 == (launch & 0x100U))
            {
                const error_t status = gpu_surface_prepare(
                    &copy->destination_surface, width, rows, &destination_layout);
                if (POUND_SUCCESS != status)
                {
                    return status;
                }
                destination_layout.sector_layout = copy->sector_layout;
                destination                      = &destination_layout;
            }
        }
        error_t status = access_rows(memory,
                                     copy->source,
                                     copy->source_pitch,
                                     source,
                                     width,
                                     rows,
                                     scratch,
                                     GPU_MEMORY_READ);
        if (POUND_SUCCESS == status)
        {
            status = access_rows(memory,
                                 copy->destination,
                                 copy->destination_pitch,
                                 destination,
                                 width,
                                 rows,
                                 scratch,
                                 0);
        }
        if (POUND_SUCCESS == status)
        {
            status = access_rows(memory,
                                 copy->destination,
                                 copy->destination_pitch,
                                 destination,
                                 width,
                                 rows,
                                 scratch,
                                 GPU_MEMORY_WRITE);
        }
        if (POUND_SUCCESS != status)
        {
            return status;
        }
    }

    if (0 != semaphore)
    {
        uint8_t bytes[4];
        for (uint32_t index = 0; index < 4U; ++index)
        {
            bytes[index] = (uint8_t)(copy->semaphore_payload >> (index * 8U));
        }
        return gpu_memory_write(memory, copy->semaphore_address, bytes, sizeof(bytes));
    }
    return POUND_SUCCESS;
}

error_t
gpu_copy_method(gpu_copy_t         *copy,
                const gpu_memory_t *memory,
                uint32_t            method,
                uint32_t            data,
                void               *scratch,
                size_t              scratch_size)
{
    if (NULL == copy || NULL == memory)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    switch (method)
    {
        case 0x100U:
            return POUND_SUCCESS;
        case 0x240U:
            return set_upper(&copy->semaphore_address, data);
        case 0x244U:
            copy->semaphore_address
                = (copy->semaphore_address & UINT64_C(0xFFFFFFFF00000000)) | data;
            break;
        case 0x248U:
            copy->semaphore_payload = data;
            break;
        case 0x300U:
            return launch_copy(copy, memory, data, scratch, scratch_size);
        case 0x400U:
            return set_upper(&copy->source, data);
        case 0x404U:
            copy->source = (copy->source & UINT64_C(0xFFFFFFFF00000000)) | data;
            break;
        case 0x408U:
            return set_upper(&copy->destination, data);
        case 0x40CU:
            copy->destination = (copy->destination & UINT64_C(0xFFFFFFFF00000000)) | data;
            break;
        case 0x410U:
            copy->source_pitch = data;
            break;
        case 0x414U:
            copy->destination_pitch = data;
            break;
        case 0x418U:
            copy->line_length = data;
            break;
        case 0x41CU:
            copy->line_count = data;
            break;
        case 0x700U:
            copy->remap_constant_a = data;
            break;
        case 0x704U:
            copy->remap_constant_b = data;
            break;
        case 0x708U:
            copy->remap_components = data;
            break;
        case 0x70CU:
            copy->destination_surface.block_size = data;
            break;
        case 0x710U:
            copy->destination_surface.width = data;
            break;
        case 0x714U:
            copy->destination_surface.height = data;
            break;
        case 0x718U:
            copy->destination_surface.depth = data;
            break;
        case 0x71CU:
            copy->destination_surface.layer = data;
            break;
        case 0x720U:
            copy->destination_surface.origin_x = data & 0xFFFFU;
            copy->destination_surface.origin_y = data >> 16U;
            break;
        case 0x728U:
            copy->source_surface.block_size = data;
            break;
        case 0x72CU:
            copy->source_surface.width = data;
            break;
        case 0x730U:
            copy->source_surface.height = data;
            break;
        case 0x734U:
            copy->source_surface.depth = data;
            break;
        case 0x738U:
            copy->source_surface.layer = data;
            break;
        case 0x73CU:
            copy->source_surface.origin_x = data & 0xFFFFU;
            copy->source_surface.origin_y = data >> 16U;
            break;
        case 0x744U:
            copy->source_surface.origin_x = data;
            break;
        case 0x748U:
            copy->source_surface.origin_y = data;
            break;
        case 0x74CU:
            copy->destination_surface.origin_x = data;
            break;
        case 0x750U:
            copy->destination_surface.origin_y = data;
            break;
        default:
            return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    return POUND_SUCCESS;
}
