#include "gpu/gpu_pushbuffer.h"
#include <inttypes.h>
#include <stdio.h>

static error_t
print_method(void *context, uint32_t subchannel, uint32_t method_address, uint32_t data)
{
    FILE *output = context;
    if (0 > fprintf(output,
                    "subchannel=%" PRIu32 " method=0x%04" PRIx32 " data=0x%08" PRIx32 "\n",
                    subchannel,
                    method_address,
                    data))
    {
        return POUND_ERROR_IO;
    }

    return POUND_SUCCESS;
}

int
main(int argc, char **argv)
{
    if (2 > argc)
    {
        fprintf(stderr, "Usage: %s <pushbuffer.bin> [next-segment.bin ...]\n", argv[0]);
        return 1;
    }

    gpu_pushbuffer_t decoder;
    if (POUND_SUCCESS != gpu_pushbuffer_init(&decoder))
    {
        return 1;
    }

    for (int argument = 1; argument < argc; ++argument)
    {
        FILE *input = fopen(argv[argument], "rb");
        if (NULL == input)
        {
            perror(argv[argument]);
            return 1;
        }

        size_t offset = 0;
        for (;;)
        {
            uint8_t      bytes[4];
            const size_t count = fread(bytes, 1, sizeof(bytes), input);
            if (0 == count)
            {
                if (0 != ferror(input))
                {
                    perror(argv[argument]);
                    fclose(input);
                    return 1;
                }
                break;
            }

            if (sizeof(bytes) != count)
            {
                fprintf(stderr, "%s: partial word at byte %zu\n", argv[argument], offset);
                fclose(input);
                return 1;
            }

            const uint32_t word = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U)
                                  | ((uint32_t)bytes[2] << 16U) | ((uint32_t)bytes[3] << 24U);
            size_t        consumed = 0;
            const error_t status   = gpu_pushbuffer_decode(
                &decoder, bytes, sizeof(bytes), print_method, stdout, &consumed);
            if (POUND_SUCCESS != status)
            {
                fprintf(stderr,
                        "%s: GPU decode error %d at byte %zu (word 0x%08" PRIx32 ")\n",
                        argv[argument],
                        (int)status,
                        offset + consumed,
                        word);
                fclose(input);
                return 1;
            }

            offset += count;
            if (0 != decoder.segment_ended)
            {
                break;
            }
        }

        if (0 != fclose(input))
        {
            perror(argv[argument]);
            return 1;
        }
    }

    if (POUND_SUCCESS != gpu_pushbuffer_finish(&decoder))
    {
        fprintf(stderr, "Incomplete packet: missing %" PRIu32 " data words\n", decoder.remaining);
        return 1;
    }

    return 0 == fflush(stdout) ? 0 : 1;
}
