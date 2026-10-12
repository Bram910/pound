#include "gpu/gpu_shader.h"
#include "gpu/gpu_shader_spirv.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHADER_TOOL_SIZE_LIMIT    (16U * 1024U * 1024U)
#define SHADER_TOOL_BUDGET_LIMIT  1000000U
#define SHADER_TOOL_PROGRAM_LIMIT 65536U

static uint8_t *
load_shader(const char *path, size_t *size)
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
    if (length <= 0 || (unsigned long)length > SHADER_TOOL_SIZE_LIMIT
        || 0 != (unsigned long)length % GPU_SHADER_INSTRUCTION_SIZE
        || 0 != fseek(input, 0, SEEK_SET))
    {
        fprintf(stderr,
                "Shader must be a nonempty sequence of 16-byte instructions, at most %u bytes\n",
                SHADER_TOOL_SIZE_LIMIT);
        fclose(input);
        return NULL;
    }
    uint8_t *bytes = malloc((size_t)length);
    if (NULL == bytes)
    {
        fclose(input);
        return NULL;
    }
    const size_t count  = fread(bytes, 1, (size_t)length, input);
    const int    closed = fclose(input);
    if ((size_t)length != count || 0 != closed)
    {
        free(bytes);
        return NULL;
    }
    *size = count;
    return bytes;
}

static int
trace_shader(const uint8_t *bytes, size_t size)
{
    for (size_t offset = 0; offset < size; offset += GPU_SHADER_INSTRUCTION_SIZE)
    {
        gpu_shader_instruction_t instruction;
        const error_t            status
            = gpu_shader_decode(bytes + offset, GPU_SHADER_INSTRUCTION_SIZE, &instruction);
        if (POUND_SUCCESS != status)
        {
            fprintf(stderr, "Shader decode error %d at byte %zu\n", (int)status, offset);
            return 1;
        }
        printf(
            "{\"offset\":%zu,\"operation\":\"%s\",\"destination\":%u,"
            "\"predicate\":%u,\"predicate_invert\":%u,\"condition\":%u,"
            "\"condition_invert\":%u,\"lookup\":%u,\"branch_bytes\":%" PRId64 ",\"sources\":[",
            offset,
            gpu_shader_operation_name(instruction.operation),
            instruction.destination,
            instruction.predicate,
            instruction.predicate_invert,
            instruction.condition,
            instruction.condition_invert,
            instruction.lookup,
            instruction.branch_bytes);
        const gpu_shader_source_t *source = instruction.sources;
        for (uint32_t index = 0; index < instruction.source_count; ++index, ++source)
        {
            printf("%s{\"kind\":\"%s\",\"value\":%u,\"negate\":%u}",
                   0 != index ? "," : "",
                   GPU_SHADER_SOURCE_REGISTER == source->kind ? "register" : "immediate",
                   source->value,
                   source->negate);
        }
        printf(
            "],\"shift_type\":%u,\"shift_right\":%u,\"shift_wrap\":%u,\"shift_high\":%u,\"return_"
            "shift\":%u,"
            "\"address_shift\":%u,\"intermediate_negate\":%u,\"carry_in\":%u,"
            "\"dot_signed\":[%u,%u],\"dot_high\":%u,\"multiply_mode\":%u,"
            "\"permute_mode\":%u,\"comparison\":%u,\"compare_signed\":%u,\"compare_extended\":%u,"
            "\"boolean_operation\":%u,"
            "\"predicate_sources\":[",
            (uint32_t)instruction.shift_type,
            instruction.shift_right,
            instruction.shift_wrap,
            instruction.shift_high,
            instruction.return_shift,
            instruction.address_shift,
            instruction.intermediate_negate,
            instruction.carry_in,
            instruction.dot_signed[0],
            instruction.dot_signed[1],
            instruction.dot_high,
            (uint32_t)instruction.multiply_mode,
            (uint32_t)instruction.permute_mode,
            (uint32_t)instruction.comparison,
            instruction.compare_signed,
            instruction.compare_extended,
            (uint32_t)instruction.boolean_operation);
        const uint32_t *predicate_source = instruction.predicate_sources;
        const uint32_t *predicate_invert = instruction.predicate_source_inverts;
        for (uint32_t index = 0; index < instruction.predicate_source_count;
             ++index, ++predicate_source, ++predicate_invert)
        {
            printf("%s{\"value\":%u,\"invert\":%u}",
                   0 != index ? "," : "",
                   *predicate_source,
                   *predicate_invert);
        }
        printf("],\"predicate_destinations\":[");
        const uint32_t *predicate_destination = instruction.predicate_destinations;
        for (uint32_t index = 0; index < instruction.predicate_destination_count;
             ++index, ++predicate_destination)
        {
            printf("%s%u", 0 != index ? "," : "", *predicate_destination);
        }
        printf(
            "],\"predicate_lookups\":[%u,%u],\"stall\":%u,\"yield\":%u,\"write_barrier\":%u,\"read_"
            "barrier\":%u,"
            "\"wait_mask\":%u,\"reuse_mask\":%u,\"words\":[%u,%u,%u,%u]}\n",
            instruction.predicate_lookups[0],
            instruction.predicate_lookups[1],
            instruction.stall,
            instruction.yield,
            instruction.write_barrier,
            instruction.read_barrier,
            instruction.wait_mask,
            instruction.reuse_mask,
            instruction.words[0],
            instruction.words[1],
            instruction.words[2],
            instruction.words[3]);
    }
    return ferror(stdout) ? 1 : 0;
}

static int
parse_budget(const char *text, uint32_t *budget)
{
    uint32_t    value  = 0;
    const char *cursor = text;
    while ('\0' != *cursor)
    {
        if (*cursor < '0' || *cursor > '9' || value > SHADER_TOOL_BUDGET_LIMIT / 10U)
        {
            return 1;
        }
        value = value * 10U + (uint32_t)(*cursor - '0');
        if (value > SHADER_TOOL_BUDGET_LIMIT)
        {
            return 1;
        }
        ++cursor;
    }
    if (0 == value)
    {
        return 1;
    }
    *budget = value;
    return 0;
}

static int
run_shader(const uint8_t *bytes, size_t size, uint32_t budget, const gpu_shader_program_t *program)
{
    gpu_shader_lane_t lane;
    if (POUND_SUCCESS != gpu_shader_lane_init(&lane))
    {
        return 1;
    }
    const error_t status = NULL == program ? gpu_shader_run(&lane, bytes, size, budget)
                                           : gpu_shader_run_prepared(&lane, program, budget);
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr,
                "Reference shader error %d at byte %zu after %" PRIu64 " steps\n",
                (int)status,
                lane.pc,
                lane.steps);
    }
    printf("{\"pc\":%zu,\"steps\":%" PRIu64 ",\"halted\":%u,\"predicates\":%u,\"registers\":{",
           lane.pc,
           lane.steps,
           lane.halted,
           lane.predicates);
    uint32_t        emitted = 0;
    const uint32_t *value   = lane.registers;
    for (uint32_t index = 0; index < GPU_SHADER_ZERO_REGISTER; ++index, ++value)
    {
        if (0 != *value)
        {
            printf("%s\"R%u\":%u", 0 != emitted ? "," : "", index, *value);
            emitted = 1U;
        }
    }
    puts("}}");
    return POUND_SUCCESS == status && 0 == ferror(stdout) ? 0 : 1;
}

static int
analyze_shader(const gpu_shader_program_t *program)
{
    printf("{\"instruction_count\":%zu,\"block_count\":%zu,\"blocks\":[",
           program->instruction_count,
           program->block_count);
    const gpu_shader_block_t *block = program->blocks;
    for (size_t index = 0; index < program->block_count; ++index, ++block)
    {
        printf("%s{\"id\":%zu,\"offset\":%zu,\"instruction_count\":%zu,\"successors\":[",
               0 != index ? "," : "",
               index,
               block->first_instruction * GPU_SHADER_INSTRUCTION_SIZE,
               block->instruction_count);
        const size_t *successor = block->successors;
        for (uint32_t edge = 0; edge < block->successor_count; ++edge, ++successor)
        {
            printf("%s%zu", 0 != edge ? "," : "", *successor);
        }
        printf("],\"exits\":%u,\"falls_off\":%u}", block->exits, block->falls_off);
    }
    puts("]}");
    return 0 == ferror(stdout) ? 0 : 1;
}

static int
write_spirv(const gpu_shader_program_t *program, const char *path, uint32_t budget)
{
    size_t  count;
    error_t status = budget ? gpu_shader_spirv_emit_bounded(program, budget, NULL, 0, &count)
                            : gpu_shader_spirv_emit(program, NULL, 0, &count);
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr, "SPIR-V emission error %d\n", (int)status);
        return 1;
    }
    uint32_t *words = malloc(count * sizeof(*words));
    if (NULL == words)
    {
        return 1;
    }
    status       = budget ? gpu_shader_spirv_emit_bounded(program, budget, words, count, &count)
                          : gpu_shader_spirv_emit(program, words, count, &count);
    FILE *output = POUND_SUCCESS == status ? fopen(path, "wb") : NULL;
    int   result = 1;
    if (NULL != output)
    {
        result               = 0;
        const uint32_t *word = words;
        for (size_t index = 0; index < count; ++index, ++word)
        {
            const uint8_t bytes[4] = { (uint8_t)*word,
                                       (uint8_t)(*word >> 8U),
                                       (uint8_t)(*word >> 16U),
                                       (uint8_t)(*word >> 24U) };
            if (sizeof(bytes) != fwrite(bytes, 1, sizeof(bytes), output))
            {
                result = 1;
                break;
            }
        }
        if (0 != fclose(output))
        {
            result = 1;
        }
    }
    free(words);
    if (result)
    {
        fprintf(stderr, "Could not write SPIR-V output\n");
    }
    return result;
}

static int
prepare_shader(const uint8_t *bytes,
               size_t         size,
               uint32_t       budget,
               uint32_t       analyze,
               const char    *output,
               uint32_t       emission_budget)
{
    const size_t count = size / GPU_SHADER_INSTRUCTION_SIZE;
    if (count > SHADER_TOOL_PROGRAM_LIMIT)
    {
        fprintf(
            stderr, "Prepared shaders are limited to %u instructions\n", SHADER_TOOL_PROGRAM_LIMIT);
        return 1;
    }
    gpu_shader_node_t  *nodes  = malloc(count * sizeof(*nodes));
    gpu_shader_block_t *blocks = malloc(count * sizeof(*blocks));
    if (NULL == nodes || NULL == blocks)
    {
        free(blocks);
        free(nodes);
        return 1;
    }
    gpu_shader_program_t program;
    const error_t        status
        = gpu_shader_program_prepare(bytes, size, nodes, count, blocks, count, &program);
    int result = 1;
    if (POUND_SUCCESS != status)
    {
        fprintf(stderr, "Shader preparation error %d\n", (int)status);
    }
    else
    {
        result = NULL != output ? write_spirv(&program, output, emission_budget)
                 : analyze      ? analyze_shader(&program)
                                : run_shader(bytes, size, budget, &program);
    }
    free(blocks);
    free(nodes);
    return result;
}

int
main(int argc, char **argv)
{
    uint32_t       budget = 100000U;
    const uint32_t run
        = argc >= 2 && (0 == strcmp(argv[1], "run") || 0 == strcmp(argv[1], "run-prepared"));
    const uint32_t bounded = argc >= 2 && 0 == strcmp(argv[1], "spirv-bounded");
    const uint32_t spirv   = bounded || (argc >= 2 && 0 == strcmp(argv[1], "spirv"));
    if (bounded)
    {
        budget = 256U;
    }
    if (argc < 3 || argc > 5
        || (!run && !spirv && 0 != strcmp(argv[1], "trace") && 0 != strcmp(argv[1], "analyze"))
        || (spirv && (argc < 4 || (!bounded && argc != 4)))
        || (!spirv && argc >= 4 && (argc != 4 || !run || 0 != parse_budget(argv[3], &budget)))
        || (argc == 5
            && (!bounded || 0 != parse_budget(argv[4], &budget)
                || budget > GPU_SHADER_SPIRV_BUDGET_LIMIT)))
    {
        fprintf(stderr,
                "Usage: %s trace|analyze <raw-sm86.bin> | run|run-prepared <raw-sm86.bin> "
                "[budget=1..1000000] | spirv <raw-sm86.bin> <output.spv> | "
                "spirv-bounded <raw-sm86.bin> <output.spv> [budget=1..4096]\n",
                argv[0]);
        return 1;
    }
    size_t   size  = 0;
    uint8_t *bytes = load_shader(argv[2], &size);
    if (NULL == bytes)
    {
        return 1;
    }
    int result;
    if (0 == strcmp(argv[1], "trace"))
    {
        result = trace_shader(bytes, size);
    }
    else if (0 == strcmp(argv[1], "run"))
    {
        result = run_shader(bytes, size, budget, NULL);
    }
    else
    {
        result = prepare_shader(bytes,
                                size,
                                budget,
                                0 == strcmp(argv[1], "analyze"),
                                spirv ? argv[3] : NULL,
                                bounded ? budget : 0);
    }
    free(bytes);
    return result;
}
