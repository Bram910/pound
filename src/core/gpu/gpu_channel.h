#ifndef POUND_GPU_CHANNEL_H
#define POUND_GPU_CHANNEL_H

#include "gpu_copy.h"
#include "gpu_inline.h"
#include "gpu_pushbuffer.h"
#include "gpu_semaphore.h"

typedef struct
{
    gpu_memory_t       *memory;
    uint8_t            *command_buffer;
    uint8_t            *transfer_buffer;
    size_t              command_capacity;
    size_t              transfer_capacity;
    size_t              pending_size;
    size_t              pending_offset;
    uint64_t            semaphore_address;
    uint64_t            semaphore_payload;
    uint64_t            methods_executed;
    uint64_t            copies_executed;
    uint64_t            uploads_executed;
    uint64_t            barriers_executed;
    gpu_clock_read_t    read_clock;
    void               *clock_context;
    gpu_copy_t          copies[8];
    gpu_inline_t        uploads[8];
    gpu_pushbuffer_t    decoder;
    uint32_t            classes[8];
    uint32_t            reference;
    uint32_t            memory_operations[3];
    uint32_t            memory_operations_ready;
    gpu_sector_layout_t sector_layout;
} gpu_channel_t;

error_t gpu_channel_init(gpu_channel_t *channel,
                         gpu_memory_t  *memory,
                         void          *command_buffer,
                         size_t         command_capacity,
                         void          *transfer_buffer,
                         size_t         transfer_capacity);
error_t gpu_channel_submit(gpu_channel_t *channel, const void *entry, size_t entry_size);
error_t gpu_channel_resume(gpu_channel_t *channel);
error_t gpu_channel_finish(const gpu_channel_t *channel);
error_t gpu_channel_reset(gpu_channel_t *channel);
error_t gpu_channel_set_sector_layout(gpu_channel_t *channel, gpu_sector_layout_t layout);
error_t gpu_channel_set_clock(gpu_channel_t   *channel,
                              gpu_clock_read_t read_clock,
                              void            *clock_context);

#endif
