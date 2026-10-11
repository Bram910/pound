#include "gpu/gpu_channel.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPLAY_MEMORY_LIMIT   (64U * 1024U * 1024U)
#define REPLAY_COMMAND_LIMIT  (8U * 1024U * 1024U)
#define REPLAY_TRANSFER_LIMIT (16U * 1024U * 1024U)

typedef struct
{
    uint64_t value;
    uint64_t step;
} replay_clock_t;

static uint64_t
read_clock(void *context)
{
    replay_clock_t *clock = context;
    clock->value += clock->step;
    return clock->value;
}

static int
parse_options(int argc, char **argv, gpu_sector_layout_t *layout, uint64_t *clock_step)
{
    if (argc < 4 || argc > 6)
    {
        return 1;
    }
    for (int index = 4; index < argc; ++index)
    {
        if (0 == strcmp(argv[index], "--sector-layout=16bx2")
            || 0 == strcmp(argv[index], "--sector-layout=turing2d"))
        {
            if (GPU_SECTOR_LAYOUT_UNSPECIFIED != *layout)
            {
                return 1;
            }
            *layout = 0 == strcmp(argv[index], "--sector-layout=16bx2")
                          ? GPU_SECTOR_LAYOUT_16BX2
                          : GPU_SECTOR_LAYOUT_TURING2D;
        }
        else if (0 == strncmp(argv[index], "--clock-step-ns=", 16U) && 0 == *clock_step)
        {
            uint64_t    value  = 0;
            const char *cursor = argv[index] + 16U;
            while ('\0' != *cursor)
            {
                if (*cursor < '0' || *cursor > '9' || value > UINT64_C(100000000))
                {
                    return 1;
                }
                value = value * 10U + (uint64_t)(*cursor - '0');
                if (value > UINT64_C(1000000000))
                {
                    return 1;
                }
                ++cursor;
            }
            if (0 == value)
            {
                return 1;
            }
            *clock_step = value;
        }
        else
        {
            return 1;
        }
    }
    return 0;
}

static uint8_t *
load_memory(const char *path, size_t *size)
{
    FILE *input = fopen(path, "rb");
    if (NULL == input)
    {
        perror(path);
        return NULL;
    }
    if (0 != fseek(input, 0, SEEK_END))
    {
        fclose(input);
        return NULL;
    }
    const long length = ftell(input);
    if (length <= 0 || (unsigned long)length > REPLAY_MEMORY_LIMIT
        || 0 != fseek(input, 0, SEEK_SET))
    {
        fprintf(stderr, "%s: memory image must contain 1 to %u bytes\n", path, REPLAY_MEMORY_LIMIT);
        fclose(input);
        return NULL;
    }
    uint8_t *bytes = malloc((size_t)length);
    if (NULL == bytes)
    {
        fclose(input);
        return NULL;
    }
    const size_t count        = fread(bytes, 1, (size_t)length, input);
    const int    close_status = fclose(input);
    if ((size_t)length != count || 0 != close_status)
    {
        free(bytes);
        return NULL;
    }
    *size = count;
    return bytes;
}

static int
save_memory(const char *path, const void *bytes, size_t size)
{
    FILE *output = fopen(path, "wb");
    if (NULL == output)
    {
        perror(path);
        return 1;
    }
    const size_t count        = fwrite(bytes, 1, size, output);
    const int    close_status = fclose(output);
    return count == size && 0 == close_status ? 0 : 1;
}

int
main(int argc, char **argv)
{
    gpu_sector_layout_t layout = GPU_SECTOR_LAYOUT_UNSPECIFIED;
    replay_clock_t      clock  = { 0 };
    if (0 != parse_options(argc, argv, &layout, &clock.step))
    {
        fprintf(stderr,
                "Usage: %s <memory.bin> <gpfifo.bin> <output-memory.bin>"
                " [--sector-layout=16bx2|turing2d] [--clock-step-ns=1..1000000000]\n",
                argv[0]);
        return 1;
    }
    size_t   size  = 0;
    uint8_t *bytes = load_memory(argv[1], &size);
    if (NULL == bytes)
    {
        return 1;
    }
    uint8_t      *commands = malloc(REPLAY_COMMAND_LIMIT);
    uint8_t      *transfer = malloc(REPLAY_TRANSFER_LIMIT);
    gpu_channel_t channel;
    gpu_memory_t  memory;
    FILE         *entries = NULL;
    int           result  = 1;

    if (NULL == commands || NULL == transfer)
    {
        fprintf(stderr, "Could not allocate GPU replay buffers\n");
        goto cleanup;
    }
    error_t status = gpu_memory_init(&memory);
    if (POUND_SUCCESS == status)
    {
        status = gpu_memory_map(&memory, 0, bytes, size, GPU_MEMORY_READ | GPU_MEMORY_WRITE);
    }
    if (POUND_SUCCESS == status)
    {
        status = gpu_channel_init(
            &channel, &memory, commands, REPLAY_COMMAND_LIMIT, transfer, REPLAY_TRANSFER_LIMIT);
    }
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr, "GPU initialization failed: %d\n", (int)status);
        goto cleanup;
    }
    if (GPU_SECTOR_LAYOUT_UNSPECIFIED != layout)
    {
        status = gpu_channel_set_sector_layout(&channel, layout);
        if (POUND_SUCCESS != status)
        {
            goto cleanup;
        }
    }
    if (0 != clock.step)
    {
        status = gpu_channel_set_clock(&channel, read_clock, &clock);
        if (POUND_SUCCESS != status)
        {
            goto cleanup;
        }
    }
    entries = fopen(argv[2], "rb");
    if (NULL == entries)
    {
        perror(argv[2]);
        goto cleanup;
    }

    size_t entry_index = 0;
    for (;;)
    {
        uint8_t      entry[8];
        const size_t count = fread(entry, 1, sizeof(entry), entries);
        if (0 == count && 0 == ferror(entries))
        {
            break;
        }
        if (sizeof(entry) != count)
        {
            fprintf(stderr, "%s: partial entry or read error at entry %zu\n", argv[2], entry_index);
            goto cleanup;
        }
        status = gpu_channel_submit(&channel, entry, sizeof(entry));
        if (POUND_SUCCESS != status)
        {
            fprintf(stderr,
                    "GPU error %d at entry %zu, segment byte %zu%s\n",
                    (int)status,
                    entry_index,
                    channel.pending_offset,
                    POUND_ERROR_GPU_WAIT == status ? " (semaphore wait)" : "");
            goto cleanup;
        }
        ++entry_index;
    }
    status = gpu_channel_finish(&channel);
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr, "Incomplete GPU stream: %d\n", (int)status);
        goto cleanup;
    }
    if (0 != fclose(entries))
    {
        entries = NULL;
        goto cleanup;
    }
    entries = NULL;
    result  = save_memory(argv[3], bytes, size);
    if (0 == result)
    {
        printf("Executed entries=%zu methods=%" PRIu64 " copies=%" PRIu64 " uploads=%" PRIu64
               " barriers=%" PRIu64 "\n",
               entry_index,
               channel.methods_executed,
               channel.copies_executed,
               channel.uploads_executed,
               channel.barriers_executed);
    }

cleanup:
    if (NULL != entries)
    {
        fclose(entries);
    }
    free(transfer);
    free(commands);
    free(bytes);
    return result;
}
