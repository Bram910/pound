#ifndef POUND_GPU_MEMORY_H
#define POUND_GPU_MEMORY_H

#include "errors.h"
#include <stddef.h>
#include <stdint.h>

#define GPU_MEMORY_MAPPING_LIMIT 128U
#define GPU_MEMORY_ADDRESS_LIMIT (UINT64_C(1) << 49U)
#define GPU_MEMORY_READ          1U
#define GPU_MEMORY_WRITE         2U

typedef struct
{
    uint64_t start;
    uint64_t end;
    uint8_t *host;
    uint32_t permissions;
    uint32_t reserved;
} gpu_mapping_t;

typedef struct
{
    gpu_mapping_t mappings[GPU_MEMORY_MAPPING_LIMIT];
    size_t        count;
} gpu_memory_t;

error_t gpu_memory_init(gpu_memory_t *memory);
error_t gpu_memory_map(
    gpu_memory_t *memory, uint64_t address, void *host, size_t size, uint32_t permissions);
error_t gpu_memory_unmap(gpu_memory_t *memory, uint64_t address, size_t size);
error_t gpu_memory_validate(const gpu_memory_t *memory,
                            uint64_t            address,
                            size_t              size,
                            uint32_t            permissions);
error_t gpu_memory_read(const gpu_memory_t *memory, uint64_t address, void *output, size_t size);
error_t gpu_memory_write(const gpu_memory_t *memory,
                         uint64_t            address,
                         const void         *input,
                         size_t              size);

#endif
