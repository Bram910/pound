#include "gpu_inline.h"
#include "gpu_semaphore.h"

static error_t
set_upper(uint64_t *address, uint32_t data)
{
    if (data > 0x1FFFFU)
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    *address = (*address & UINT64_C(0xFFFFFFFF)) | ((uint64_t)data << 32U);
    return POUND_SUCCESS;
}

static error_t
validate_semaphore(const gpu_inline_t *upload, const gpu_memory_t *memory, uint32_t flags)
{
    if (2U != ((flags >> 4U) & 3U))
    {
        return POUND_SUCCESS;
    }
    if (0 != (upload->semaphore_address & 3U))
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    return gpu_memory_validate(memory, upload->semaphore_address, 4U, GPU_MEMORY_WRITE);
}

static error_t
release_semaphore(const gpu_inline_t *upload, const gpu_memory_t *memory, uint32_t flags)
{
    return 2U == ((flags >> 4U) & 3U)
               ? gpu_semaphore_execute(
                     memory, upload->semaphore_address, upload->semaphore_payload, 1U)
               : POUND_SUCCESS;
}

static error_t
launch_upload(gpu_inline_t *upload, const gpu_memory_t *memory, uint32_t flags, size_t capacity)
{
    const uint32_t completion = (flags >> 4U) & 3U;
    if (0 != (flags & ~0x1071U) || completion > 2U || (2U == completion && 0 == (flags & 0x1000U)))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    if (0 != (upload->line_length & 3U))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    const uint64_t total = (uint64_t)upload->line_length * upload->line_count;
    if (total > capacity)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    error_t status = validate_semaphore(upload, memory, flags);
    if (POUND_SUCCESS != status)
    {
        return status;
    }
    gpu_surface_layout_t layout = { 0 };
    if (0 != total && 0 == (flags & 1U))
    {
        if (GPU_SECTOR_LAYOUT_16BX2 != upload->sector_layout
            && GPU_SECTOR_LAYOUT_TURING2D != upload->sector_layout)
        {
            return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
        }
        status = gpu_surface_prepare(
            &upload->surface, upload->line_length, upload->line_count, &layout);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        layout.sector_layout = upload->sector_layout;
    }
    if (0 != total && upload->destination >= GPU_MEMORY_ADDRESS_LIMIT)
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    for (uint32_t row = 0; row < upload->line_count && 0 != upload->line_length; ++row)
    {
        uint32_t column = 0;
        while (column < upload->line_length)
        {
            const uint64_t offset = 0 != (flags & 1U) ? (uint64_t)row * upload->pitch + column
                                                      : gpu_surface_offset(&layout, column, row);
            if (offset >= GPU_MEMORY_ADDRESS_LIMIT - upload->destination)
            {
                return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
            }
            uint32_t length = upload->line_length - column;
            if (0 == (flags & 1U))
            {
                const uint32_t available = 16U - ((layout.origin_x + column) & 15U);
                if (length > available)
                {
                    length = available;
                }
            }
            status = gpu_memory_validate(
                memory, upload->destination + offset, length, GPU_MEMORY_WRITE);
            if (POUND_SUCCESS != status)
            {
                return status;
            }
            column += length;
        }
    }
    if (0 == total)
    {
        status = release_semaphore(upload, memory, flags);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
    }
    upload->layout        = layout;
    upload->launch_flags  = flags;
    upload->bytes_written = 0;
    upload->total_bytes   = total;
    upload->active        = 0 != total;
    return POUND_SUCCESS;
}

static error_t
load_word(gpu_inline_t *upload, const gpu_memory_t *memory, uint32_t data)
{
    if (0 == upload->active)
    {
        return POUND_ERROR_GPU_INVALID_COMMAND;
    }
    const uint32_t row    = (uint32_t)(upload->bytes_written / upload->line_length);
    const uint32_t column = (uint32_t)(upload->bytes_written % upload->line_length);
    const uint32_t final  = upload->total_bytes - upload->bytes_written == 4U;
    if (0 != final)
    {
        const error_t status = validate_semaphore(upload, memory, upload->launch_flags);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
    }
    uint64_t addresses[2];
    uint32_t lengths[2];
    uint32_t count    = 0;
    uint32_t consumed = 0;
    uint8_t  bytes[4];
    for (uint32_t index = 0; index < 4U; ++index)
    {
        bytes[index] = (uint8_t)(data >> (index * 8U));
    }
    while (consumed < 4U)
    {
        const uint64_t offset = 0 != (upload->launch_flags & 1U)
                                    ? (uint64_t)row * upload->pitch + column + consumed
                                    : gpu_surface_offset(&upload->layout, column + consumed, row);
        uint32_t       length = 4U - consumed;
        if (0 == (upload->launch_flags & 1U))
        {
            const uint32_t available = 16U - ((upload->layout.origin_x + column + consumed) & 15U);
            if (length > available)
            {
                length = available;
            }
        }
        addresses[count] = upload->destination + offset;
        lengths[count]   = length;
        const error_t status
            = gpu_memory_validate(memory, addresses[count], length, GPU_MEMORY_WRITE);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        ++count;
        consumed += length;
    }
    consumed = 0;
    for (uint32_t index = 0; index < count; ++index)
    {
        const error_t status
            = gpu_memory_write(memory, addresses[index], bytes + consumed, lengths[index]);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        consumed += lengths[index];
    }
    if (0 != final)
    {
        const error_t status = release_semaphore(upload, memory, upload->launch_flags);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        upload->active = 0;
    }
    upload->bytes_written += 4U;
    return POUND_SUCCESS;
}

error_t
gpu_inline_method(gpu_inline_t       *upload,
                  const gpu_memory_t *memory,
                  uint32_t            method,
                  uint32_t            data,
                  size_t              transfer_capacity)
{
    if (NULL == upload || NULL == memory)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (0x100U == method)
    {
        return POUND_SUCCESS;
    }
    if (0x1B4U == method)
    {
        return load_word(upload, memory, data);
    }
    if (0 != upload->active)
    {
        return POUND_ERROR_GPU_BUSY;
    }
    switch (method)
    {
        case 0x110U:
            return 0 == data ? POUND_SUCCESS : POUND_ERROR_GPU_INVALID_COMMAND;
        case 0x180U:
            upload->line_length = data;
            break;
        case 0x184U:
            upload->line_count = data;
            break;
        case 0x188U:
            return set_upper(&upload->destination, data);
        case 0x18CU:
            upload->destination = (upload->destination & UINT64_C(0xFFFFFFFF00000000)) | data;
            break;
        case 0x190U:
            upload->pitch = data;
            break;
        case 0x194U:
            if (0 != (data & ~0xFFFU))
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            upload->surface.block_size = data | 0x1000U;
            break;
        case 0x198U:
            upload->surface.width = data;
            break;
        case 0x19CU:
            upload->surface.height = data;
            break;
        case 0x1A0U:
            upload->surface.depth = data;
            break;
        case 0x1A4U:
            upload->surface.layer = data;
            break;
        case 0x1A8U:
            if (data > 0x1FFFFFU)
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            upload->surface.origin_x = data;
            break;
        case 0x1ACU:
            if (data > 0x1FFFFU)
            {
                return POUND_ERROR_GPU_INVALID_COMMAND;
            }
            upload->surface.origin_y = data;
            break;
        case 0x1B0U:
            return launch_upload(upload, memory, data, transfer_capacity);
        case 0x1DCU:
            return set_upper(&upload->semaphore_address, data);
        case 0x1E0U:
            upload->semaphore_address
                = (upload->semaphore_address & UINT64_C(0xFFFFFFFF00000000)) | data;
            break;
        case 0x1E4U:
            upload->semaphore_payload = data;
            break;
        default:
            return POUND_ERROR_GPU_UNSUPPORTED_COMMAND;
    }
    return POUND_SUCCESS;
}
