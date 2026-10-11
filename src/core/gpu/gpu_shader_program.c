#include "gpu_shader.h"

static error_t
branch_target(const gpu_shader_instruction_t *instruction,
              size_t                          offset,
              size_t                          size,
              size_t                         *target)
{
    size_t next = offset + GPU_SHADER_INSTRUCTION_SIZE;
    if (instruction->branch_bytes < 0)
    {
        const uint64_t distance = (uint64_t)-instruction->branch_bytes;
        if (distance > next)
        {
            return POUND_ERROR_GPU_INVALID_SHADER;
        }
        next -= (size_t)distance;
    }
    else
    {
        const uint64_t distance = (uint64_t)instruction->branch_bytes;
        if (distance >= size - next)
        {
            return POUND_ERROR_GPU_INVALID_SHADER;
        }
        next += (size_t)distance;
    }
    if (next >= size)
    {
        return POUND_ERROR_GPU_INVALID_SHADER;
    }
    *target = next / GPU_SHADER_INSTRUCTION_SIZE;
    return POUND_SUCCESS;
}

static uint32_t
predicate_can_be(uint32_t predicate, uint32_t invert, uint32_t value)
{
    return GPU_SHADER_TRUE_PREDICATE != predicate || (1U ^ invert) == value;
}

error_t
gpu_shader_program_prepare(const void           *bytes,
                           size_t                size,
                           gpu_shader_node_t    *nodes,
                           size_t                node_capacity,
                           gpu_shader_block_t   *blocks,
                           size_t                block_capacity,
                           gpu_shader_program_t *program)
{
    if (NULL == bytes || NULL == nodes || NULL == blocks || NULL == program || 0 == size
        || 0 != size % GPU_SHADER_INSTRUCTION_SIZE)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const size_t count = size / GPU_SHADER_INSTRUCTION_SIZE;
    if (count > node_capacity || count > block_capacity || count > SIZE_MAX / sizeof(*nodes)
        || count > SIZE_MAX / sizeof(*blocks))
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    const uint8_t *cursor = bytes;
    for (size_t index = 0; index < count; ++index, cursor += GPU_SHADER_INSTRUCTION_SIZE)
    {
        gpu_shader_instruction_t instruction;
        const error_t status = gpu_shader_decode(cursor, GPU_SHADER_INSTRUCTION_SIZE, &instruction);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        if (GPU_SHADER_BRA == instruction.operation)
        {
            size_t        target;
            const error_t branch_status
                = branch_target(&instruction, index * GPU_SHADER_INSTRUCTION_SIZE, size, &target);
            if (POUND_SUCCESS != branch_status)
            {
                return branch_status;
            }
        }
    }
    cursor                  = bytes;
    gpu_shader_node_t *node = nodes;
    for (size_t index = 0; index < count; ++index, ++node, cursor += GPU_SHADER_INSTRUCTION_SIZE)
    {
        gpu_shader_decode(cursor, GPU_SHADER_INSTRUCTION_SIZE, &node->instruction);
        node->block_index        = SIZE_MAX;
        node->branch_instruction = SIZE_MAX;
        if (GPU_SHADER_BRA == node->instruction.operation)
        {
            branch_target(&node->instruction,
                          index * GPU_SHADER_INSTRUCTION_SIZE,
                          size,
                          &node->branch_instruction);
        }
    }
    nodes[0].block_index = 0;
    node                 = nodes;
    for (size_t index = 0; index < count; ++index, ++node)
    {
        const gpu_shader_operation_t operation = node->instruction.operation;
        if (GPU_SHADER_BRA == operation)
        {
            nodes[node->branch_instruction].block_index = 0;
        }
        if ((GPU_SHADER_BRA == operation || GPU_SHADER_EXIT == operation) && index + 1U < count)
        {
            nodes[index + 1U].block_index = 0;
        }
    }
    size_t block_count = 0;
    node               = nodes;
    for (size_t index = 0; index < count; ++index, ++node)
    {
        if (0 == node->block_index)
        {
            blocks[block_count++] = (gpu_shader_block_t) { .first_instruction = index };
        }
        node->block_index = block_count - 1U;
    }
    gpu_shader_block_t *block = blocks;
    for (size_t index = 0; index < block_count; ++index, ++block)
    {
        const size_t end = index + 1U < block_count ? blocks[index + 1U].first_instruction : count;
        block->instruction_count                    = end - block->first_instruction;
        const gpu_shader_node_t        *last        = nodes + end - 1U;
        const gpu_shader_instruction_t *instruction = &last->instruction;
        const uint32_t                  can_execute
            = predicate_can_be(instruction->predicate, instruction->predicate_invert, 1U);
        const uint32_t can_skip
            = predicate_can_be(instruction->predicate, instruction->predicate_invert, 0);
        uint32_t fallthrough     = 1U;
        uint32_t successor_count = 0;
        if (GPU_SHADER_BRA == instruction->operation)
        {
            if (can_execute
                && predicate_can_be(instruction->condition, instruction->condition_invert, 1U)
                && !(instruction->predicate == instruction->condition
                     && instruction->predicate_invert != instruction->condition_invert))
            {
                block->successors[successor_count++] = nodes[last->branch_instruction].block_index;
            }
            fallthrough
                = can_skip
                  || predicate_can_be(instruction->condition, instruction->condition_invert, 0);
        }
        else if (GPU_SHADER_EXIT == instruction->operation)
        {
            block->exits = can_execute;
            fallthrough  = can_skip;
        }
        if (fallthrough)
        {
            if (end == count)
            {
                block->falls_off = 1U;
            }
            else if (0 == successor_count || block->successors[0] != index + 1U)
            {
                block->successors[successor_count++] = index + 1U;
            }
        }
        block->successor_count = successor_count;
    }
    *program = (gpu_shader_program_t) { .nodes             = nodes,
                                        .blocks            = blocks,
                                        .instruction_count = count,
                                        .block_count       = block_count,
                                        .byte_size         = size };
    return POUND_SUCCESS;
}
