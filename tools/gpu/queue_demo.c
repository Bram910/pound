#include "gpu/gpu_queue.h"
#include <stdio.h>
#include <string.h>

static void
write_word(uint8_t *bytes, uint32_t value)
{
    for (uint32_t index = 0; index < 4U; ++index)
    {
        bytes[index] = (uint8_t)(value >> (index * 8U));
    }
}

static size_t
method(uint8_t *bytes, size_t offset, uint32_t subchannel, uint32_t address, uint32_t value)
{
    write_word(bytes + offset, 0x20010000U | (subchannel << 13U) | (address / 4U));
    write_word(bytes + offset + 4U, value);
    return offset + 8U;
}

static void
entry(uint8_t *bytes, uint32_t address, size_t size)
{
    write_word(bytes, address);
    write_word(bytes + 4U, (uint32_t)(size / 4U) << 10U);
}

static int
check(uint32_t condition, const char *description)
{
    if (0 == condition)
    {
        fprintf(stderr, "Queue demo failed: %s\n", description);
        return 0;
    }
    return 1;
}

int
main(void)
{
    uint8_t       backing[4096];
    uint8_t       commands[512];
    uint8_t       transfer[256];
    uint8_t       producer_commands[64];
    uint8_t       producer_transfer[16];
    gpu_memory_t  memory;
    gpu_channel_t channel;
    gpu_channel_t producer;
    gpu_queue_t   queue;
    memset(backing, 0xA5, sizeof(backing));
    memset(backing + 2312U, 0, 8U);
    if (!check(POUND_SUCCESS == gpu_memory_init(&memory), "mapper initialization")
        || !check(POUND_SUCCESS
                      == gpu_memory_map(
                          &memory, 0, backing, sizeof(backing), GPU_MEMORY_READ | GPU_MEMORY_WRITE),
                  "mapping")
        || !check(
            POUND_SUCCESS
                == gpu_channel_init(
                    &channel, &memory, commands, sizeof(commands), transfer, sizeof(transfer)),
            "consumer initialization")
        || !check(POUND_SUCCESS
                      == gpu_channel_init(&producer,
                                          &memory,
                                          producer_commands,
                                          sizeof(producer_commands),
                                          producer_transfer,
                                          sizeof(producer_transfer)),
                  "producer initialization")
        || !check(POUND_SUCCESS == gpu_queue_init(&queue, &channel, 128U, 8U, 6U),
                  "ring initialization"))
    {
        return 1;
    }

    size_t offset = 0;
    offset        = method(backing + 256U, offset, 0, 0, GPU_CLASS_KEPLER_INLINE_TO_MEMORY);
    offset        = method(backing + 256U, offset, 0, 0x180U, 16U);
    offset        = method(backing + 256U, offset, 0, 0x184U, 1U);
    offset        = method(backing + 256U, offset, 0, 0x18CU, 2048U);
    offset        = method(backing + 256U, offset, 0, 0x190U, 16U);
    offset        = method(backing + 256U, offset, 0, 0x1E0U, 2304U);
    offset        = method(backing + 256U, offset, 0, 0x1E4U, 7U);
    offset        = method(backing + 256U, offset, 0, 0x1B0U, 0x1021U);
    write_word(backing + 256U + offset, 0x6004006DU);
    write_word(backing + 256U + offset + 4U, 0x03020100U);
    write_word(backing + 256U + offset + 8U, 0x07060504U);
    entry(backing + 128U + 6U * 8U, 256U, offset + 12U);

    write_word(backing + 512U, 0x0B0A0908U);
    write_word(backing + 516U, 0x0F0E0D0CU);
    offset = 8U;
    offset = method(backing + 512U, offset, 0, 0x5CU, 2304U);
    offset = method(backing + 512U, offset, 0, 0x64U, 5U);
    offset = method(backing + 512U, offset, 0, 0x6CU, 2U);
    entry(backing + 128U + 7U * 8U, 512U, offset);

    offset = 0;
    offset = method(backing + 768U, offset, 0, 0x5CU, 2312U);
    offset = method(backing + 768U, offset, 0, 0x64U, 1U);
    offset = method(backing + 768U, offset, 0, 0x6CU, 0U);
    offset = method(backing + 768U, offset, 0, 0x50U, 99U);
    entry(backing + 128U, 768U, offset);

    offset                = 0;
    offset                = method(backing + 896U, offset, 4U, 0, GPU_CLASS_AMPERE_DMA_COPY);
    const uint32_t copy[] = { 0, 2048U, 0, 2080U, 16U, 16U, 16U, 1U };
    for (uint32_t index = 0; index < 8U; ++index)
    {
        offset = method(backing + 896U, offset, 4U, 0x400U + index * 4U, copy[index]);
    }
    offset = method(backing + 896U, offset, 4U, 0x300U, 0x181U);
    entry(backing + 128U + 8U, 896U, offset);

    uint32_t completed = 0;
    if (!check(POUND_SUCCESS == gpu_queue_drain(&queue, 2U, 1U, &completed) && 1U == completed
                   && 7U == queue.get && 1U == channel.uploads[0].active,
               "budgeted upload launch and split packet")
        || !check(POUND_ERROR_GPU_WAIT == gpu_queue_drain(&queue, 2U, 8U, &completed)
                      && 1U == completed && 0 == queue.get && 1U == queue.pending
                      && 1U == channel.uploads_executed && 0 == channel.copies_executed,
                  "wrapped ring and unsatisfied acquire"))
    {
        return 1;
    }
    const uint64_t methods_before = channel.methods_executed;
    if (!check(POUND_ERROR_GPU_WAIT == gpu_queue_drain(&queue, 2U, 8U, &completed) && 0 == completed
                   && channel.methods_executed == methods_before,
               "wait retry without duplicate execution"))
    {
        return 1;
    }

    write_word(backing + 768U + 28U, 0xDEADBEEFU);
    offset = 0;
    offset = method(backing + 1024U, offset, 0, 0x5CU, 2312U);
    offset = method(backing + 1024U, offset, 0, 0x64U, 1U);
    offset = method(backing + 1024U, offset, 0, 0x6CU, 1U);
    uint8_t producer_entry[8];
    entry(producer_entry, 1024U, offset);
    if (!check(POUND_SUCCESS
                   == gpu_channel_submit(&producer, producer_entry, sizeof(producer_entry)),
               "producer semaphore release")
        || !check(POUND_SUCCESS == gpu_queue_drain(&queue, 2U, 8U, &completed) && 2U == completed
                      && 2U == queue.get && 0 == queue.pending && 99U == channel.reference
                      && 1U == channel.copies_executed && 1U == channel.uploads_executed,
                  "resume preserved snapshot and execute dependent copy")
        || !check(POUND_SUCCESS == gpu_channel_finish(&channel), "completed stream"))
    {
        return 1;
    }
    for (uint32_t index = 0; index < 16U; ++index)
    {
        if (!check(backing[2048U + index] == index && backing[2080U + index] == index,
                   "uploaded and copied bytes"))
        {
            return 1;
        }
    }
    if (!check(7U == backing[2304U] && 0 == backing[2305U] && 0 == backing[2306U]
                   && 0 == backing[2307U] && 0xA5U == backing[2079U] && 0xA5U == backing[2096U],
               "completion semaphore and untouched padding"))
    {
        return 1;
    }
    puts(
        "Verified: wrapped GPFIFO ring, budgeted draining, split inline upload, cross-channel "
        "semaphore,");
    puts("wait/resume without duplicates, command snapshot preservation and dependent DMA copy.");
    return 0;
}
