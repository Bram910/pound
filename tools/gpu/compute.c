#include "gpu/gpu_shader_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COMPUTE_CHAIN_DISPATCH_LIMIT 4096U

static int
parse_number(const char *text, uint32_t *output)
{
    if (NULL == text || '\0' == *text)
    {
        return 0;
    }
    uint32_t value = 0;
    while ('\0' != *text)
    {
        if (*text < '0' || *text > '9')
        {
            return 0;
        }
        const uint32_t digit = (uint32_t)(*text++ - '0');
        if (value > (UINT32_MAX - digit) / 10U)
        {
            return 0;
        }
        value = value * 10U + digit;
    }
    *output = value;
    return 1;
}

static uint8_t *
load_file(const char *path, size_t limit, size_t *length)
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
    const long size = ftell(input);
    if (size <= 0 || (unsigned long)size > limit || 0 != fseek(input, 0, SEEK_SET))
    {
        fclose(input);
        return NULL;
    }
    uint8_t *bytes = malloc((size_t)size);
    if (NULL == bytes)
    {
        fclose(input);
        return NULL;
    }
    const size_t read_size = fread(bytes, 1, (size_t)size, input);
    const int    closed    = fclose(input);
    if (read_size != (size_t)size || 0 != closed)
    {
        free(bytes);
        return NULL;
    }
    *length = read_size;
    return bytes;
}

static int
execute(const char *code_path,
        const char *input_path,
        const char *output_path,
        uint32_t    budget,
        uint32_t    device,
        uint32_t    max_dispatches)
{
    const uint32_t stride
        = budget == 0 ? GPU_SHADER_SPIRV_LANE_WORDS : GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS;
    size_t   code_size  = 0;
    size_t   input_size = 0;
    uint8_t *code       = load_file(
        code_path, GPU_SHADER_SPIRV_INSTRUCTION_LIMIT * GPU_SHADER_INSTRUCTION_SIZE, &code_size);
    uint8_t *input = load_file(
        input_path, GPU_SHADER_VULKAN_LANE_LIMIT * stride * sizeof(uint32_t), &input_size);
    int                          result  = 1;
    gpu_shader_node_t           *nodes   = NULL;
    gpu_shader_block_t          *blocks  = NULL;
    uint32_t                    *state   = NULL;
    gpu_shader_vulkan_session_t *session = NULL;
    gpu_shader_vulkan_kernel_t  *kernel  = NULL;
    if (NULL == code || NULL == input || code_size % GPU_SHADER_INSTRUCTION_SIZE != 0
        || input_size % (stride * sizeof(uint32_t)) != 0)
    {
        fprintf(stderr, "Invalid shader or lane buffer size\n");
        goto cleanup;
    }
    const size_t count       = code_size / GPU_SHADER_INSTRUCTION_SIZE;
    const size_t state_words = input_size / sizeof(uint32_t);
    nodes                    = malloc(count * sizeof(*nodes));
    blocks                   = malloc(count * sizeof(*blocks));
    state                    = malloc(input_size);
    if (NULL == nodes || NULL == blocks || NULL == state)
    {
        goto cleanup;
    }
    gpu_shader_program_t program;
    error_t              status
        = gpu_shader_program_prepare(code, code_size, nodes, count, blocks, count, &program);
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr, "Shader preparation error %d\n", (int)status);
        goto cleanup;
    }
    for (size_t index = 0; index < state_words; ++index)
    {
        const uint8_t *word = input + index * 4U;
        state[index] = (uint32_t)word[0] | ((uint32_t)word[1] << 8U) | ((uint32_t)word[2] << 16U)
                       | ((uint32_t)word[3] << 24U);
    }
    gpu_shader_vulkan_report_t report;
    if (max_dispatches == 0)
    {
        status = gpu_shader_vulkan_run(&program, budget, state, state_words, device, &report);
    }
    else
    {
        status
            = gpu_shader_vulkan_session_create(device, (uint32_t)(state_words / stride), &session);
        if (status == POUND_SUCCESS)
        {
            status = gpu_shader_vulkan_kernel_create(session, &program, budget, &kernel);
        }
        for (uint32_t index = 0; status == POUND_SUCCESS && index < max_dispatches; ++index)
        {
            status = gpu_shader_vulkan_dispatch(session, kernel, state, state_words, &report);
            if (status != POUND_SUCCESS)
            {
                break;
            }
            uint32_t retryable = 0;
            for (size_t offset = 0; offset < state_words; offset += stride)
            {
                const uint32_t overflow
                    = state[offset + GPU_SHADER_SPIRV_STEPS_LOW] == UINT32_MAX
                      && state[offset + GPU_SHADER_SPIRV_STEPS_HIGH] == UINT32_MAX;
                retryable
                    |= state[offset + GPU_SHADER_SPIRV_STATUS] == POUND_ERROR_GPU_RESOURCE_LIMIT
                       && overflow == 0 && state[offset + GPU_SHADER_SPIRV_HALTED] == 0;
            }
            if (retryable == 0)
            {
                break;
            }
        }
    }
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr, "Vulkan execution error %d; output file was not opened\n", (int)status);
        goto cleanup;
    }
    for (size_t index = 0; index < state_words; ++index)
    {
        uint8_t *word = input + index * 4U;
        for (uint32_t byte = 0; byte < 4U; ++byte)
        {
            word[byte] = (uint8_t)(state[index] >> (byte * 8U));
        }
    }
    FILE *output = fopen(output_path, "wb");
    if (NULL == output)
    {
        perror(output_path);
        goto cleanup;
    }
    const size_t written = fwrite(input, 1, input_size, output);
    const int    closed  = fclose(output);
    if (written != input_size || 0 != closed)
    {
        fprintf(stderr, "Failed to write lane output\n");
        goto cleanup;
    }
    printf("%s (device %u): %u independent lanes, %u words/lane, %s host memory\n",
           report.device_name,
           report.device_index,
           report.lane_count,
           report.lane_words,
           report.coherent_memory ? "coherent" : "flushed/invalidate");
    if (max_dispatches != 0)
    {
        printf("Reused one session and compiled kernel for %llu dispatches (cap %u)\n",
               (unsigned long long)report.dispatch_index,
               max_dispatches);
    }
    if (budget != 0)
    {
        size_t unfinished = 0;
        size_t faults     = 0;
        for (size_t offset = GPU_SHADER_SPIRV_STATUS; offset < state_words; offset += stride)
        {
            unfinished += state[offset] == POUND_ERROR_GPU_RESOURCE_LIMIT;
            faults += state[offset] != POUND_SUCCESS
                      && state[offset] != POUND_ERROR_GPU_RESOURCE_LIMIT;
        }
        printf(
            "Lane results: %zu success, %zu unfinished/overflow, %zu faults; status is stored at "
            "word 260\n",
            state_words / stride - unfinished - faults,
            unfinished,
            faults);
    }
    result = 0;

cleanup:
    gpu_shader_vulkan_kernel_destroy(kernel);
    if (session != NULL)
    {
        gpu_shader_vulkan_session_destroy(session);
    }
    free(state);
    free(blocks);
    free(nodes);
    free(input);
    free(code);
    return result;
}

int
main(int argc, char **argv)
{
    const uint32_t chain           = argc >= 2 && 0 == strcmp(argv[1], "bounded-chain");
    const uint32_t bounded         = chain || (argc >= 2 && 0 == strcmp(argv[1], "bounded"));
    uint32_t       budget          = bounded ? 256U : 0U;
    uint32_t       device          = GPU_SHADER_VULKAN_AUTO_DEVICE;
    uint32_t       max_dispatches  = chain ? 256U : 0U;
    const int      device_argument = chain ? 7 : (bounded ? 6 : 5);
    if (argc < 5 || argc > device_argument + 1 || (!bounded && 0 != strcmp(argv[1], "straight"))
        || (bounded && argc >= 6
            && (!parse_number(argv[5], &budget) || budget == 0
                || budget > GPU_SHADER_SPIRV_BUDGET_LIMIT))
        || (chain && argc >= 7
            && (!parse_number(argv[6], &max_dispatches) || max_dispatches == 0
                || max_dispatches > COMPUTE_CHAIN_DISPATCH_LIMIT))
        || (argc > device_argument && 0 != strcmp(argv[device_argument], "auto")
            && (!parse_number(argv[device_argument], &device)
                || device == GPU_SHADER_VULKAN_AUTO_DEVICE)))
    {
        fprintf(stderr,
                "Usage: %s straight <raw-sm86.bin> <lane-input.bin> <lane-output.bin> "
                "[device=auto|index]\n"
                "       %s bounded <raw-sm86.bin> <lane-input.bin> <lane-output.bin> "
                "[budget=1..4096] [device=auto|index]\n"
                "       %s bounded-chain <raw-sm86.bin> <lane-input.bin> <lane-output.bin> "
                "[budget=1..4096] [max-dispatches=1..4096] [device=auto|index]\n",
                argv[0],
                argv[0],
                argv[0]);
        return 1;
    }
    return execute(argv[2], argv[3], argv[4], budget, device, max_dispatches);
}
