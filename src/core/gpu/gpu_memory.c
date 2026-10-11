#include "gpu_memory.h"
#include <string.h>

static int
valid_range(uint64_t address, size_t size)
{
    return address <= GPU_MEMORY_ADDRESS_LIMIT && size <= GPU_MEMORY_ADDRESS_LIMIT - address;
}

static inline const gpu_mapping_t *
find_mapping(const gpu_memory_t *memory, uint64_t address)
{
    if (memory->count <= 1U)
    {
        const gpu_mapping_t *mapping = memory->mappings;
        return 0 != memory->count && address >= mapping->start && address < mapping->end ? mapping
                                                                                         : NULL;
    }
    if (memory->count <= 16U)
    {
        const gpu_mapping_t *cursor = memory->mappings;
        for (size_t index = 0; index < memory->count; ++index, ++cursor)
        {
            if (address < cursor->start)
            {
                break;
            }
            if (address < cursor->end)
            {
                return cursor;
            }
        }
        return NULL;
    }
    size_t lower = 0;
    size_t upper = memory->count;
    while (lower < upper)
    {
        const size_t         middle  = lower + (upper - lower) / 2U;
        const gpu_mapping_t *mapping = memory->mappings + middle;
        if (address < mapping->start)
        {
            upper = middle;
        }
        else if (address >= mapping->end)
        {
            lower = middle + 1U;
        }
        else
        {
            return mapping;
        }
    }
    return NULL;
}

error_t
gpu_memory_init(gpu_memory_t *memory)
{
    if (NULL == memory)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    memset(memory, 0, sizeof(*memory));
    return POUND_SUCCESS;
}

error_t
gpu_memory_map(
    gpu_memory_t *memory, uint64_t address, void *host, size_t size, uint32_t permissions)
{
    if (NULL == memory || NULL == host || 0 == size || 0 == permissions
        || 0 != (permissions & ~(GPU_MEMORY_READ | GPU_MEMORY_WRITE))
        || size > UINTPTR_MAX - (uintptr_t)host)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (!valid_range(address, size))
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    if (memory->count >= GPU_MEMORY_MAPPING_LIMIT)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }

    const uint64_t end       = address + size;
    size_t         insertion = 0;
    while (insertion < memory->count && memory->mappings[insertion].start < address)
    {
        ++insertion;
    }
    if ((0 != insertion && memory->mappings[insertion - 1U].end > address)
        || (insertion < memory->count && memory->mappings[insertion].start < end))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }

    memmove(memory->mappings + insertion + 1U,
            memory->mappings + insertion,
            (memory->count - insertion) * sizeof(gpu_mapping_t));
    memory->mappings[insertion] = (gpu_mapping_t) { address, end, host, permissions, 0U };
    ++memory->count;
    return POUND_SUCCESS;
}

error_t
gpu_memory_unmap(gpu_memory_t *memory, uint64_t address, size_t size)
{
    if (NULL == memory || 0 == size)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (!valid_range(address, size))
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }

    for (size_t index = 0; index < memory->count; ++index)
    {
        if (memory->mappings[index].start == address
            && memory->mappings[index].end == address + size)
        {
            memmove(memory->mappings + index,
                    memory->mappings + index + 1U,
                    (memory->count - index - 1U) * sizeof(gpu_mapping_t));
            --memory->count;
            memset(memory->mappings + memory->count, 0, sizeof(gpu_mapping_t));
            return POUND_SUCCESS;
        }
    }
    return POUND_ERROR_GUEST_ADDRESS_OUT_OF_BOUNDS;
}

error_t
gpu_memory_validate(const gpu_memory_t *memory, uint64_t address, size_t size, uint32_t permissions)
{
    if (NULL == memory || 0 == permissions
        || 0 != (permissions & ~(GPU_MEMORY_READ | GPU_MEMORY_WRITE)))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (!valid_range(address, size))
    {
        return POUND_ERROR_GUEST_ADDRESS_OVERFLOW;
    }
    if (0 == size)
    {
        return POUND_SUCCESS;
    }

    const uint64_t       end     = address + size;
    uint64_t             cursor  = address;
    const gpu_mapping_t *mapping = find_mapping(memory, cursor);
    if (NULL == mapping)
    {
        return POUND_ERROR_GUEST_ADDRESS_OUT_OF_BOUNDS;
    }
    const gpu_mapping_t *mapping_end = memory->mappings + memory->count;
    while (cursor < end)
    {
        if (mapping == mapping_end || cursor < mapping->start)
        {
            return POUND_ERROR_GUEST_ADDRESS_OUT_OF_BOUNDS;
        }
        if ((mapping->permissions & permissions) != permissions)
        {
            return POUND_ERROR_MEMORY_FAULT;
        }
        cursor = mapping->end < end ? mapping->end : end;
        ++mapping;
    }
    return POUND_SUCCESS;
}

error_t
gpu_memory_read(const gpu_memory_t *memory, uint64_t address, void *output, size_t size)
{
    if (NULL == output && 0 != size)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const error_t status = gpu_memory_validate(memory, address, size, GPU_MEMORY_READ);
    if (POUND_SUCCESS != status || 0 == size)
    {
        return status;
    }

    const gpu_mapping_t *mapping   = find_mapping(memory, address);
    uint8_t             *cursor    = output;
    size_t               remaining = size;
    uint64_t             current   = address;
    while (0 != remaining)
    {
        const size_t available = (size_t)(mapping->end - current);
        const size_t chunk     = remaining < available ? remaining : available;
        memcpy(cursor, mapping->host + (size_t)(current - mapping->start), chunk);
        cursor += chunk;
        current += chunk;
        remaining -= chunk;
        ++mapping;
    }
    return POUND_SUCCESS;
}

error_t
gpu_memory_write(const gpu_memory_t *memory, uint64_t address, const void *input, size_t size)
{
    if (NULL == input && 0 != size)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const error_t status = gpu_memory_validate(memory, address, size, GPU_MEMORY_WRITE);
    if (POUND_SUCCESS != status || 0 == size)
    {
        return status;
    }

    const gpu_mapping_t *mapping   = find_mapping(memory, address);
    const uint8_t       *cursor    = input;
    size_t               remaining = size;
    uint64_t             current   = address;
    while (0 != remaining)
    {
        const size_t available = (size_t)(mapping->end - current);
        const size_t chunk     = remaining < available ? remaining : available;
        memcpy(mapping->host + (size_t)(current - mapping->start), cursor, chunk);
        cursor += chunk;
        current += chunk;
        remaining -= chunk;
        ++mapping;
    }
    return POUND_SUCCESS;
}
