#ifndef POUND_GPU_SHADER_H
#define POUND_GPU_SHADER_H

#include "errors.h"
#include <stddef.h>
#include <stdint.h>

#define GPU_SHADER_INSTRUCTION_SIZE 16U
#define GPU_SHADER_ZERO_REGISTER    255U
#define GPU_SHADER_TRUE_PREDICATE   7U

typedef enum
{
    GPU_SHADER_MOV,
    GPU_SHADER_IADD3,
    GPU_SHADER_IMAD,
    GPU_SHADER_LOP3,
    GPU_SHADER_POPC,
    GPU_SHADER_SEL,
    GPU_SHADER_NOP,
    GPU_SHADER_BRA,
    GPU_SHADER_EXIT,
    GPU_SHADER_ISETP,
    GPU_SHADER_PLOP3,
    GPU_SHADER_IMNMX,
    GPU_SHADER_BREV,
    GPU_SHADER_IABS,
    GPU_SHADER_SHF,
    GPU_SHADER_FLO,
    GPU_SHADER_PRMT,
    GPU_SHADER_SGXT,
    GPU_SHADER_BMSK,
    GPU_SHADER_LEA,
    GPU_SHADER_IDP4,
    GPU_SHADER_IDP2,
} gpu_shader_operation_t;

typedef enum
{
    GPU_SHADER_PERMUTE_INDEX,
    GPU_SHADER_PERMUTE_FORWARD4,
    GPU_SHADER_PERMUTE_BACKWARD4,
    GPU_SHADER_PERMUTE_REPLICATE8,
    GPU_SHADER_PERMUTE_CLAMP_LEFT,
    GPU_SHADER_PERMUTE_CLAMP_RIGHT,
    GPU_SHADER_PERMUTE_REPLICATE16,
} gpu_shader_permute_mode_t;

typedef enum
{
    GPU_SHADER_SHIFT_S64,
    GPU_SHADER_SHIFT_U64,
    GPU_SHADER_SHIFT_S32,
    GPU_SHADER_SHIFT_U32,
} gpu_shader_shift_type_t;

typedef enum
{
    GPU_SHADER_COMPARE_FALSE,
    GPU_SHADER_COMPARE_LT,
    GPU_SHADER_COMPARE_EQ,
    GPU_SHADER_COMPARE_LE,
    GPU_SHADER_COMPARE_GT,
    GPU_SHADER_COMPARE_NE,
    GPU_SHADER_COMPARE_GE,
    GPU_SHADER_COMPARE_TRUE,
} gpu_shader_comparison_t;

typedef enum
{
    GPU_SHADER_BOOLEAN_AND,
    GPU_SHADER_BOOLEAN_OR,
    GPU_SHADER_BOOLEAN_XOR,
} gpu_shader_boolean_t;

typedef enum
{
    GPU_SHADER_SOURCE_REGISTER,
    GPU_SHADER_SOURCE_IMMEDIATE,
} gpu_shader_source_kind_t;

typedef enum
{
    GPU_SHADER_MULTIPLY_LOW,
    GPU_SHADER_MULTIPLY_WIDE,
    GPU_SHADER_MULTIPLY_HIGH,
} gpu_shader_multiply_t;

typedef struct
{
    gpu_shader_source_kind_t kind;
    uint32_t                 value;
    uint32_t                 negate;
} gpu_shader_source_t;

typedef struct
{
    gpu_shader_operation_t    operation;
    gpu_shader_source_t       sources[3];
    uint32_t                  words[4];
    int64_t                   branch_bytes;
    uint32_t                  source_count;
    uint32_t                  destination;
    uint32_t                  predicate;
    uint32_t                  predicate_invert;
    uint32_t                  condition;
    uint32_t                  condition_invert;
    uint32_t                  lookup;
    gpu_shader_comparison_t   comparison;
    gpu_shader_boolean_t      boolean_operation;
    uint32_t                  compare_signed;
    uint32_t                  compare_extended;
    gpu_shader_shift_type_t   shift_type;
    uint32_t                  shift_right;
    uint32_t                  shift_wrap;
    uint32_t                  shift_high;
    uint32_t                  return_shift;
    uint32_t                  address_shift;
    uint32_t                  intermediate_negate;
    uint32_t                  carry_in;
    uint32_t                  dot_signed[2];
    uint32_t                  dot_high;
    gpu_shader_multiply_t     multiply_mode;
    gpu_shader_permute_mode_t permute_mode;
    uint32_t                  predicate_destinations[2];
    uint32_t                  predicate_sources[3];
    uint32_t                  predicate_source_inverts[3];
    uint32_t                  predicate_lookups[2];
    uint32_t                  predicate_source_count;
    uint32_t                  predicate_destination_count;
    uint32_t                  stall;
    uint32_t                  yield;
    uint32_t                  write_barrier;
    uint32_t                  read_barrier;
    uint32_t                  wait_mask;
    uint32_t                  reuse_mask;
} gpu_shader_instruction_t;

typedef struct
{
    uint32_t registers[GPU_SHADER_ZERO_REGISTER];
    uint32_t predicates;
    uint32_t halted;
    size_t   pc;
    uint64_t steps;
} gpu_shader_lane_t;

typedef struct
{
    gpu_shader_instruction_t instruction;
    size_t                   block_index;
    size_t                   branch_instruction;
} gpu_shader_node_t;

typedef struct
{
    size_t   first_instruction;
    size_t   instruction_count;
    size_t   successors[2];
    uint32_t successor_count;
    uint32_t exits;
    uint32_t falls_off;
} gpu_shader_block_t;

typedef struct
{
    const gpu_shader_node_t  *nodes;
    const gpu_shader_block_t *blocks;
    size_t                    instruction_count;
    size_t                    block_count;
    size_t                    byte_size;
} gpu_shader_program_t;

error_t gpu_shader_decode(const void *bytes, size_t size, gpu_shader_instruction_t *instruction);
const char *gpu_shader_operation_name(gpu_shader_operation_t operation);
error_t     gpu_shader_lane_init(gpu_shader_lane_t *lane);
error_t gpu_shader_run(gpu_shader_lane_t *lane, const void *program, size_t size, uint32_t budget);
error_t gpu_shader_program_prepare(const void           *bytes,
                                   size_t                size,
                                   gpu_shader_node_t    *nodes,
                                   size_t                node_capacity,
                                   gpu_shader_block_t   *blocks,
                                   size_t                block_capacity,
                                   gpu_shader_program_t *program);
error_t gpu_shader_run_prepared(gpu_shader_lane_t          *lane,
                                const gpu_shader_program_t *program,
                                uint32_t                    budget);

#endif
