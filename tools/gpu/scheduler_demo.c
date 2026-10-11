#include "gpu/gpu_scheduler.h"
#include <inttypes.h>
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

static uint64_t
read_value(const uint8_t *bytes)
{
    uint64_t value = 0;
    for (uint32_t index = 0; index < 8U; ++index)
    {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static size_t
method(uint8_t *bytes, size_t offset, uint32_t subchannel, uint32_t address, uint32_t value)
{
    write_word(bytes + offset, 0x20010000U | (subchannel << 13U) | (address / 4U));
    write_word(bytes + offset + 4U, value);
    return offset + 8U;
}

static size_t
semaphore(uint8_t *bytes, size_t offset, uint32_t address, uint32_t value, uint32_t flags)
{
    offset = method(bytes, offset, 0, 0x5CU, address);
    offset = method(bytes, offset, 0, 0x64U, value);
    return method(bytes, offset, 0, 0x6CU, flags);
}

static size_t
barrier(uint8_t *bytes, size_t offset, uint32_t type)
{
    offset = method(bytes, offset, 0, 0x28U, 0);
    offset = method(bytes, offset, 0, 0x2CU, 0);
    offset = method(bytes, offset, 0, 0x30U, type);
    return method(bytes, offset, 0, 0x34U, 0x28000000U);
}

static size_t
copy(uint8_t *bytes, size_t offset, uint32_t source, uint32_t destination)
{
    offset                   = method(bytes, offset, 4U, 0, GPU_CLASS_AMPERE_DMA_COPY);
    const uint32_t  values[] = { 0, source, 0, destination, 16U, 16U, 16U, 1U };
    const uint32_t *cursor   = values;
    for (uint32_t index = 0; index < 8U; ++index, ++cursor)
    {
        offset = method(bytes, offset, 4U, 0x400U + index * 4U, *cursor);
    }
    return method(bytes, offset, 4U, 0x300U, 0x181U);
}

static uint64_t
read_clock(void *context)
{
    uint64_t *time = context;
    *time += 100U;
    return *time;
}

static int
check(uint32_t condition, const char *description)
{
    if (0 == condition)
    {
        fprintf(stderr, "Scheduler demo failed: %s\n", description);
        return 0;
    }
    return 1;
}

int
main(void)
{
    uint8_t                backing[8192];
    uint8_t                commands[4][512];
    uint8_t                transfers[4][64];
    gpu_channel_t          channels[4];
    gpu_queue_t            queues[4];
    gpu_queue_t           *attached[4];
    gpu_scheduler_t        scheduler;
    gpu_scheduler_result_t result;
    gpu_memory_t           memory;
    uint64_t               time = 0;
    memset(backing, 0xA5, sizeof(backing));
    memset(backing + 4608U, 0, 48U);
    if (!check(POUND_SUCCESS == gpu_memory_init(&memory)
                   && POUND_SUCCESS
                          == gpu_memory_map(&memory,
                                            0,
                                            backing,
                                            sizeof(backing),
                                            GPU_MEMORY_READ | GPU_MEMORY_WRITE),
               "shared memory"))
    {
        return 1;
    }
    for (uint32_t index = 0; index < 4U; ++index)
    {
        if (!check(POUND_SUCCESS
                           == gpu_channel_init(&channels[index],
                                               &memory,
                                               commands[index],
                                               sizeof(commands[index]),
                                               transfers[index],
                                               sizeof(transfers[index]))
                       && POUND_SUCCESS
                              == gpu_channel_set_clock(&channels[index], read_clock, &time)
                       && POUND_SUCCESS
                              == gpu_queue_init(
                                  &queues[index], &channels[index], 128U + index * 64U, 8U, 7U),
                   "channel and wrapped ring initialization"))
        {
            return 1;
        }
        attached[index] = &queues[index];
    }
    size_t offset = semaphore(backing + 1024U, 0, 4624U, 2U, 0);
    offset        = copy(backing + 1024U, offset, 4224U, 4352U);
    offset        = barrier(backing + 1024U, offset, 1U);
    offset        = semaphore(backing + 1024U, offset, 4640U, 3U, 0x02100001U);
    write_word(backing + 128U + 56U, 1024U);
    write_word(backing + 128U + 60U, (uint32_t)(offset / 4U) << 10U);

    offset = semaphore(backing + 1536U, 0, 4608U, 1U, 0);
    offset = copy(backing + 1536U, offset, 4096U, 4224U);
    offset = barrier(backing + 1536U, offset, 0);
    offset = semaphore(backing + 1536U, offset, 4624U, 2U, 0x02100001U);
    write_word(backing + 192U + 56U, 1536U);
    write_word(backing + 192U + 60U, (uint32_t)(offset / 4U) << 10U);

    offset = method(backing + 2048U, 0, 0, 0, GPU_CLASS_KEPLER_INLINE_TO_MEMORY);
    offset = method(backing + 2048U, offset, 0, 0x180U, 16U);
    offset = method(backing + 2048U, offset, 0, 0x184U, 1U);
    offset = method(backing + 2048U, offset, 0, 0x18CU, 4096U);
    offset = method(backing + 2048U, offset, 0, 0x190U, 16U);
    offset = method(backing + 2048U, offset, 0, 0x1B0U, 1U);
    for (uint32_t index = 0; index < 4U; ++index)
    {
        offset = method(backing + 2048U, offset, 0, 0x1B4U, 0x03020100U + index * 0x04040404U);
    }
    offset = barrier(backing + 2048U, offset, 0);
    offset = semaphore(backing + 2048U, offset, 4608U, 1U, 0x02100001U);
    write_word(backing + 256U + 56U, 2048U);
    write_word(backing + 256U + 60U, (uint32_t)(offset / 4U) << 10U);
    write_word(backing + 320U + 56U, 8192U);
    write_word(backing + 320U + 60U, 2U << 10U);
    if (!check(POUND_SUCCESS == gpu_scheduler_init(&scheduler, attached, 4U), "scheduler attach"))
    {
        return 1;
    }
    for (uint32_t index = 0; index < 4U; ++index)
    {
        if (!check(POUND_SUCCESS == gpu_scheduler_publish(&scheduler, index, 0), "producer PUT"))
        {
            return 1;
        }
    }
    if (!check(POUND_SUCCESS == gpu_scheduler_run(&scheduler, 2U, &result) && 2U == result.attempted
                   && 0 == result.completed && 2U == result.waiting && 1U == result.budget_exhausted
                   && 0 == time,
               "bounded waits and round-robin continuation"))
    {
        return 1;
    }
    write_word(backing + 1024U + 12U, 0xDEADBEEFU);
    const error_t fault = gpu_scheduler_run(&scheduler, 32U, &result);
    if (!check(POUND_ERROR_GUEST_ADDRESS_OUT_OF_BOUNDS == fault && 3U == result.completed
                   && 1U == result.faulted && 3U == result.fault_slot && 1U == result.pending
                   && 7U == queues[3].get && 300U == time,
               "fault isolation, dependency resolution and timestamp ordering"))
    {
        return 1;
    }
    for (uint32_t index = 0; index < 3U; ++index)
    {
        if (!check(0 == queues[index].get && POUND_SUCCESS == gpu_channel_finish(&channels[index])
                       && 1U == channels[index].barriers_executed,
                   "dependent queues drained and barriers completed"))
        {
            return 1;
        }
    }
    for (uint32_t index = 0; index < 16U; ++index)
    {
        if (!check(index == backing[4096U + index] && index == backing[4224U + index]
                       && index == backing[4352U + index],
                   "upload and two-stage copy"))
        {
            return 1;
        }
    }
    for (uint32_t index = 0; index < 3U; ++index)
    {
        if (!check(index + 1U == read_value(backing + 4608U + index * 16U)
                       && (index + 1U) * 100U == read_value(backing + 4616U + index * 16U),
                   "32-bit payload padding and 64-bit completion clock"))
        {
            return 1;
        }
    }
    uint8_t repaired[8] = { 0 };
    if (!check(
            POUND_SUCCESS
                    == gpu_memory_map(&memory, 8192U, repaired, sizeof(repaired), GPU_MEMORY_READ)
                && POUND_SUCCESS == gpu_scheduler_run(&scheduler, 8U, &result)
                && 1U == result.completed && 0 == result.pending && 300U == time,
            "repair and retry without duplicated completions"))
    {
        return 1;
    }
    printf(
        "Verified: four-channel scheduling, wrapped rings, budgeted waits, fault isolation,\n"
        "snapshot preservation, inline upload, dependent DMA, GPU/system barriers and\n"
        "timestamped completion records at 100, 200 and %" PRIu64 " ns.\n",
        time);
    return 0;
}
