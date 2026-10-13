#include "gpu_shader_spirv.h"

#define CONSTANT_LIMIT (GPU_SHADER_SPIRV_INSTRUCTION_LIMIT * 2U + 286U)

typedef enum
{
    SPIRV_VOID = 1,
    SPIRV_UINT,
    SPIRV_INT,
    SPIRV_BOOL,
    SPIRV_VECTOR,
    SPIRV_ARRAY,
    SPIRV_BUFFER_TYPE,
    SPIRV_BUFFER_POINTER,
    SPIRV_WORD_POINTER,
    SPIRV_INPUT_POINTER,
    SPIRV_FUNCTION_TYPE,
    SPIRV_BUFFER,
    SPIRV_GLOBAL_ID,
    SPIRV_MAIN,
    SPIRV_FALSE,
    SPIRV_TRUE,
    SPIRV_ENTRY,
    SPIRV_PAIR,
    SPIRV_LOCAL_WORD_POINTER,
    SPIRV_LOCAL_BOOL_POINTER,
} spirv_id_t;

typedef enum
{
    SPIRV_MEMORY_MODEL           = 14,
    SPIRV_ENTRY_POINT            = 15,
    SPIRV_EXECUTION_MODE         = 16,
    SPIRV_CAPABILITY             = 17,
    SPIRV_TYPE_VOID              = 19,
    SPIRV_TYPE_BOOL              = 20,
    SPIRV_TYPE_INT               = 21,
    SPIRV_TYPE_VECTOR            = 23,
    SPIRV_TYPE_RUNTIME_ARRAY     = 29,
    SPIRV_TYPE_STRUCT            = 30,
    SPIRV_TYPE_POINTER           = 32,
    SPIRV_TYPE_FUNCTION          = 33,
    SPIRV_CONSTANT_TRUE          = 41,
    SPIRV_CONSTANT_FALSE         = 42,
    SPIRV_CONSTANT               = 43,
    SPIRV_FUNCTION               = 54,
    SPIRV_FUNCTION_END           = 56,
    SPIRV_VARIABLE               = 59,
    SPIRV_LOAD                   = 61,
    SPIRV_STORE                  = 62,
    SPIRV_ACCESS_CHAIN           = 65,
    SPIRV_DECORATE               = 71,
    SPIRV_MEMBER_DECORATE        = 72,
    SPIRV_EXTRACT                = 81,
    SPIRV_BITCAST                = 124,
    SPIRV_ADD                    = 128,
    SPIRV_SUBTRACT               = 130,
    SPIRV_MULTIPLY               = 132,
    SPIRV_ADD_CARRY              = 149,
    SPIRV_SUBTRACT_BORROW        = 150,
    SPIRV_MULTIPLY_UNSIGNED_WIDE = 151,
    SPIRV_MULTIPLY_SIGNED_WIDE   = 152,
    SPIRV_BOOL_EQUAL             = 164,
    SPIRV_BOOL_NOT_EQUAL         = 165,
    SPIRV_BOOL_OR                = 166,
    SPIRV_BOOL_AND               = 167,
    SPIRV_BOOL_NOT               = 168,
    SPIRV_SELECT                 = 169,
    SPIRV_EQUAL                  = 170,
    SPIRV_NOT_EQUAL              = 171,
    SPIRV_UNSIGNED_GREATER       = 172,
    SPIRV_SIGNED_GREATER         = 173,
    SPIRV_UNSIGNED_GREATER_EQUAL = 174,
    SPIRV_SIGNED_GREATER_EQUAL   = 175,
    SPIRV_UNSIGNED_LESS          = 176,
    SPIRV_SIGNED_LESS            = 177,
    SPIRV_UNSIGNED_LESS_EQUAL    = 178,
    SPIRV_SIGNED_LESS_EQUAL      = 179,
    SPIRV_SHIFT_RIGHT            = 194,
    SPIRV_SHIFT_RIGHT_SIGNED     = 195,
    SPIRV_SHIFT_LEFT             = 196,
    SPIRV_OR                     = 197,
    SPIRV_XOR                    = 198,
    SPIRV_AND                    = 199,
    SPIRV_NOT                    = 200,
    SPIRV_FIELD_SIGNED           = 202,
    SPIRV_FIELD_UNSIGNED         = 203,
    SPIRV_REVERSE                = 204,
    SPIRV_COUNT                  = 205,
    SPIRV_LOOP_MERGE             = 246,
    SPIRV_SELECTION_MERGE        = 247,
    SPIRV_LABEL                  = 248,
    SPIRV_BRANCH                 = 249,
    SPIRV_BRANCH_CONDITIONAL     = 250,
    SPIRV_SWITCH                 = 251,
    SPIRV_RETURN                 = 253,
} spirv_opcode_t;

typedef struct
{
    uint32_t *words;
    size_t    count;
    uint32_t  next_id;
    uint32_t  constants[CONSTANT_LIMIT];
    uint32_t  constant_count;
    uint32_t  budget;
    uint32_t  registers[GPU_SHADER_ZERO_REGISTER];
    uint32_t  pointers[GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS];
    uint32_t  predicates[GPU_SHADER_TRUE_PREDICATE];
    uint32_t  register_variables[GPU_SHADER_ZERO_REGISTER];
    uint32_t  predicate_variables[GPU_SHADER_TRUE_PREDICATE];
    uint32_t  state_variables[6];
    uint32_t  counter_variable;
} spirv_builder_t;

typedef struct
{
    uint32_t low;
    uint32_t high;
} spirv_wide_t;

#define EMIT(builder, opcode, ...)           \
    emit(builder,                            \
         opcode,                             \
         (const uint32_t[]) { __VA_ARGS__ }, \
         sizeof((const uint32_t[]) { __VA_ARGS__ }) / sizeof(uint32_t))

static void
emit(spirv_builder_t *builder, uint32_t opcode, const uint32_t *operands, size_t count)
{
    if (NULL != builder->words)
    {
        uint32_t *cursor = builder->words + builder->count;
        *cursor++        = (uint32_t)((count + 1U) << 16U) | opcode;
        for (size_t index = 0; index < count; ++index, ++operands)
        {
            *cursor++ = *operands;
        }
    }
    builder->count += count + 1U;
}

static uint32_t
constant(const spirv_builder_t *builder, uint32_t value)
{
    const uint32_t *cursor = builder->constants;
    for (uint32_t index = 0; index < builder->constant_count; ++index, ++cursor)
    {
        if (*cursor == value)
        {
            return SPIRV_LOCAL_BOOL_POINTER + 1U + index;
        }
    }
    return 0;
}

static uint32_t
unary(spirv_builder_t *builder, uint32_t opcode, uint32_t type, uint32_t source)
{
    const uint32_t result = builder->next_id++;
    EMIT(builder, opcode, type, result, source);
    return result;
}

static uint32_t
binary(spirv_builder_t *builder, uint32_t opcode, uint32_t type, uint32_t first, uint32_t second)
{
    const uint32_t result = builder->next_id++;
    EMIT(builder, opcode, type, result, first, second);
    return result;
}

static uint32_t
select_value(
    spirv_builder_t *builder, uint32_t type, uint32_t condition, uint32_t first, uint32_t second)
{
    const uint32_t result = builder->next_id++;
    EMIT(builder, SPIRV_SELECT, type, result, condition, first, second);
    return result;
}

static uint32_t
select_word(spirv_builder_t *builder, uint32_t condition, uint32_t first, uint32_t second)
{
    const uint32_t mask = builder->next_id++;
    EMIT(builder,
         SPIRV_SELECT,
         SPIRV_UINT,
         mask,
         condition,
         constant(builder, UINT32_MAX),
         constant(builder, 0));
    return binary(builder,
                  SPIRV_XOR,
                  SPIRV_UINT,
                  second,
                  binary(builder,
                         SPIRV_AND,
                         SPIRV_UINT,
                         mask,
                         binary(builder, SPIRV_XOR, SPIRV_UINT, first, second)));
}

static uint32_t
extract(spirv_builder_t *builder, uint32_t value, uint32_t member)
{
    const uint32_t result = builder->next_id++;
    EMIT(builder, SPIRV_EXTRACT, SPIRV_UINT, result, value, member);
    return result;
}

static uint32_t
predicate(spirv_builder_t *builder, uint32_t source, uint32_t invert)
{
    const uint32_t value
        = source == GPU_SHADER_TRUE_PREDICATE ? SPIRV_TRUE : builder->predicates[source];
    return invert ? unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, value) : value;
}

static uint32_t
combine(spirv_builder_t *builder, uint32_t first, uint32_t second, gpu_shader_boolean_t operation)
{
    const uint32_t opcode = operation == GPU_SHADER_BOOLEAN_AND  ? SPIRV_BOOL_AND
                            : operation == GPU_SHADER_BOOLEAN_OR ? SPIRV_BOOL_OR
                                                                 : SPIRV_BOOL_NOT_EQUAL;
    return binary(builder, opcode, SPIRV_BOOL, first, second);
}

static uint32_t
compare(spirv_builder_t        *builder,
        uint32_t                first,
        uint32_t                second,
        gpu_shader_comparison_t operation,
        uint32_t                signed_value)
{
    if (operation == GPU_SHADER_COMPARE_FALSE || operation == GPU_SHADER_COMPARE_TRUE)
    {
        return operation == GPU_SHADER_COMPARE_FALSE ? SPIRV_FALSE : SPIRV_TRUE;
    }
    if (signed_value)
    {
        first  = unary(builder, SPIRV_BITCAST, SPIRV_INT, first);
        second = unary(builder, SPIRV_BITCAST, SPIRV_INT, second);
    }
    uint32_t opcode;
    switch (operation)
    {
        case GPU_SHADER_COMPARE_EQ:
            opcode = SPIRV_EQUAL;
            break;
        case GPU_SHADER_COMPARE_NE:
            opcode = SPIRV_NOT_EQUAL;
            break;
        case GPU_SHADER_COMPARE_LT:
            opcode = signed_value ? SPIRV_SIGNED_LESS : SPIRV_UNSIGNED_LESS;
            break;
        case GPU_SHADER_COMPARE_LE:
            opcode = signed_value ? SPIRV_SIGNED_LESS_EQUAL : SPIRV_UNSIGNED_LESS_EQUAL;
            break;
        case GPU_SHADER_COMPARE_GT:
            opcode = signed_value ? SPIRV_SIGNED_GREATER : SPIRV_UNSIGNED_GREATER;
            break;
        default:
            opcode = signed_value ? SPIRV_SIGNED_GREATER_EQUAL : SPIRV_UNSIGNED_GREATER_EQUAL;
            break;
    }
    return binary(builder, opcode, SPIRV_BOOL, first, second);
}

static uint32_t
source(const spirv_builder_t *builder, const gpu_shader_source_t *operand)
{
    if (operand->kind == GPU_SHADER_SOURCE_IMMEDIATE)
    {
        return constant(builder, operand->value);
    }
    return operand->value == GPU_SHADER_ZERO_REGISTER ? constant(builder, 0)
                                                      : builder->registers[operand->value];
}

static uint32_t
float_source(spirv_builder_t *builder, const gpu_shader_source_t *operand, uint32_t ftz)
{
    uint32_t value = source(builder, operand);
    if (operand->absolute)
    {
        value = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x7FFFFFFFU));
    }
    if (operand->negate)
    {
        value = binary(builder, SPIRV_XOR, SPIRV_UINT, value, constant(builder, 0x80000000U));
    }
    if (ftz)
    {
        const uint32_t exponent
            = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x7F800000U));
        value = select_value(
            builder,
            SPIRV_UINT,
            binary(builder, SPIRV_EQUAL, SPIRV_BOOL, exponent, constant(builder, 0)),
            binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x80000000U)),
            value);
    }
    return value;
}

static uint32_t
float_nan(spirv_builder_t *builder, uint32_t value)
{
    return compare(builder,
                   binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x7FFFFFFFU)),
                   constant(builder, 0x7F800000U),
                   GPU_SHADER_COMPARE_GT,
                   0);
}

static uint32_t
float_key(spirv_builder_t *builder, uint32_t value)
{
    const uint32_t sign
        = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x80000000U));
    return select_value(
        builder,
        SPIRV_UINT,
        binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, sign, constant(builder, 0)),
        unary(builder, SPIRV_NOT, SPIRV_UINT, value),
        binary(builder, SPIRV_OR, SPIRV_UINT, value, constant(builder, 0x80000000U)));
}

static uint32_t
float_zero(spirv_builder_t *builder, uint32_t value)
{
    const uint32_t magnitude
        = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x7FFFFFFFU));
    return select_value(builder,
                        SPIRV_UINT,
                        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, magnitude, constant(builder, 0)),
                        constant(builder, 0),
                        value);
}

static uint32_t
compare_float(spirv_builder_t              *builder,
              uint32_t                      first,
              uint32_t                      second,
              gpu_shader_float_comparison_t operation)
{
    const uint32_t nan = binary(
        builder, SPIRV_BOOL_OR, SPIRV_BOOL, float_nan(builder, first), float_nan(builder, second));
    const uint32_t relation = compare(builder,
                                      float_key(builder, float_zero(builder, first)),
                                      float_key(builder, float_zero(builder, second)),
                                      (gpu_shader_comparison_t)(operation & 7U),
                                      0);
    return (operation & 8U) != 0 ? binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, relation, nan)
                                 : binary(builder,
                                          SPIRV_BOOL_AND,
                                          SPIRV_BOOL,
                                          relation,
                                          unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, nan));
}

static uint32_t
float_minmax(spirv_builder_t *builder, uint32_t first, uint32_t second, uint32_t minimum)
{
    const uint32_t left_nan  = float_nan(builder, first);
    const uint32_t right_nan = float_nan(builder, second);
    const uint32_t less      = compare(
        builder, float_key(builder, first), float_key(builder, second), GPU_SHADER_COMPARE_LT, 0);
    uint32_t value = select_value(builder,
                                  SPIRV_UINT,
                                  binary(builder, SPIRV_BOOL_EQUAL, SPIRV_BOOL, less, minimum),
                                  first,
                                  second);
    value          = select_value(builder, SPIRV_UINT, right_nan, first, value);
    value          = select_value(
        builder,
        SPIRV_UINT,
        left_nan,
        select_value(builder, SPIRV_UINT, right_nan, constant(builder, 0x7FFFFFFFU), second),
        value);
    return value;
}

static uint32_t
float_leading_zeros(spirv_builder_t *builder, uint32_t value)
{
    for (uint32_t distance = 1U; distance < 32U; distance *= 2U)
    {
        value = binary(
            builder,
            SPIRV_OR,
            SPIRV_UINT,
            value,
            binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, distance)));
    }
    return binary(builder,
                  SPIRV_SUBTRACT,
                  SPIRV_UINT,
                  constant(builder, 32),
                  unary(builder, SPIRV_COUNT, SPIRV_UINT, value));
}

static uint32_t
float_shift_jam(spirv_builder_t *builder, uint32_t value, uint32_t distance)
{
    const uint32_t zero   = constant(builder, 0);
    const uint32_t amount = binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31));
    const uint32_t mask
        = binary(builder,
                 SPIRV_SUBTRACT,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, constant(builder, 1), amount),
                 constant(builder, 1));
    const uint32_t lost = binary(builder,
                                 SPIRV_NOT_EQUAL,
                                 SPIRV_BOOL,
                                 binary(builder, SPIRV_AND, SPIRV_UINT, value, mask),
                                 zero);
    const uint32_t shifted
        = binary(builder,
                 SPIRV_OR,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, amount),
                 select_value(builder, SPIRV_UINT, lost, constant(builder, 1), zero));
    return select_value(builder,
                        SPIRV_UINT,
                        compare(builder, distance, constant(builder, 32), GPU_SHADER_COMPARE_GE, 0),
                        select_value(builder,
                                     SPIRV_UINT,
                                     binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, value, zero),
                                     constant(builder, 1),
                                     zero),
                        shifted);
}

static uint32_t
float_mantissa(spirv_builder_t *builder, uint32_t value, uint32_t *exponent)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t raw_exponent
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, 23)),
                 constant(builder, 255));
    const uint32_t normal = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, raw_exponent, zero);
    *exponent = select_value(builder, SPIRV_UINT, normal, raw_exponent, constant(builder, 1));
    return binary(builder,
                  SPIRV_OR,
                  SPIRV_UINT,
                  binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x007FFFFFU)),
                  select_value(builder, SPIRV_UINT, normal, constant(builder, 0x00800000U), zero));
}

static uint32_t
float_round(spirv_builder_t                *builder,
            uint32_t                        significand,
            uint32_t                        exponent,
            uint32_t                        sign,
            const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t one  = constant(builder, 1);
    const uint32_t tiny = compare(builder, exponent, one, GPU_SHADER_COMPARE_LT, 1);
    significand         = select_value(
        builder,
        SPIRV_UINT,
        tiny,
        float_shift_jam(
            builder, significand, binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, one, exponent)),
        significand);
    exponent = select_value(builder, SPIRV_UINT, tiny, one, exponent);
    const uint32_t discarded
        = binary(builder, SPIRV_AND, SPIRV_UINT, significand, constant(builder, 7));
    uint32_t mantissa
        = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, significand, constant(builder, 3));
    const uint32_t negative = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, sign, zero);
    uint32_t       normal
        = compare(builder, significand, constant(builder, 0x04000000U), GPU_SHADER_COMPARE_GE, 0);
    if (instruction->float_rounding == GPU_SHADER_ROUND_NEAREST)
    {
        normal = compare(
            builder, significand, constant(builder, 0x03FFFFFEU), GPU_SHADER_COMPARE_GE, 0);
    }
    else if (instruction->float_rounding != GPU_SHADER_ROUND_ZERO)
    {
        const uint32_t outward = instruction->float_rounding == GPU_SHADER_ROUND_DOWN
                                     ? negative
                                     : unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative);
        normal                 = select_value(
            builder,
            SPIRV_BOOL,
            outward,
            compare(builder, significand, constant(builder, 0x03FFFFFCU), GPU_SHADER_COMPARE_GT, 0),
            normal);
    }
    const uint32_t flush     = binary(builder,
                                  SPIRV_BOOL_AND,
                                  SPIRV_BOOL,
                                  binary(builder, SPIRV_EQUAL, SPIRV_BOOL, exponent, one),
                                  unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, normal));
    uint32_t       increment = SPIRV_FALSE;
    if (instruction->float_rounding == GPU_SHADER_ROUND_NEAREST)
    {
        increment = binary(
            builder,
            SPIRV_BOOL_OR,
            SPIRV_BOOL,
            compare(builder, discarded, constant(builder, 4), GPU_SHADER_COMPARE_GT, 0),
            binary(builder,
                   SPIRV_BOOL_AND,
                   SPIRV_BOOL,
                   binary(builder, SPIRV_EQUAL, SPIRV_BOOL, discarded, constant(builder, 4)),
                   binary(builder,
                          SPIRV_NOT_EQUAL,
                          SPIRV_BOOL,
                          binary(builder, SPIRV_AND, SPIRV_UINT, mantissa, one),
                          zero)));
    }
    else if (instruction->float_rounding != GPU_SHADER_ROUND_ZERO)
    {
        increment = binary(builder,
                           SPIRV_BOOL_AND,
                           SPIRV_BOOL,
                           binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, discarded, zero),
                           instruction->float_rounding == GPU_SHADER_ROUND_DOWN
                               ? negative
                               : unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative));
    }
    mantissa = binary(builder,
                      SPIRV_ADD,
                      SPIRV_UINT,
                      mantissa,
                      select_value(builder, SPIRV_UINT, increment, one, zero));
    const uint32_t carry
        = compare(builder, mantissa, constant(builder, 0x01000000U), GPU_SHADER_COMPARE_GE, 0);
    mantissa                      = select_value(builder,
                            SPIRV_UINT,
                            carry,
                            binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, mantissa, one),
                            mantissa);
    exponent                      = binary(builder,
                      SPIRV_ADD,
                      SPIRV_UINT,
                      exponent,
                      select_value(builder, SPIRV_UINT, carry, one, zero));
    const uint32_t exponent_field = select_value(
        builder,
        SPIRV_UINT,
        compare(builder, mantissa, constant(builder, 0x00800000U), GPU_SHADER_COMPARE_GE, 0),
        binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, exponent, constant(builder, 23)),
        zero);
    uint32_t value = binary(
        builder,
        SPIRV_OR,
        SPIRV_UINT,
        sign,
        binary(builder,
               SPIRV_OR,
               SPIRV_UINT,
               exponent_field,
               binary(builder, SPIRV_AND, SPIRV_UINT, mantissa, constant(builder, 0x007FFFFFU))));
    const uint32_t infinity = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST ? SPIRV_TRUE
                              : instruction->float_rounding == GPU_SHADER_ROUND_ZERO  ? SPIRV_FALSE
                              : instruction->float_rounding == GPU_SHADER_ROUND_DOWN
                                  ? negative
                                  : unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative);
    value
        = select_value(builder,
                       SPIRV_UINT,
                       compare(builder, exponent, constant(builder, 255), GPU_SHADER_COMPARE_GE, 0),
                       binary(builder,
                              SPIRV_OR,
                              SPIRV_UINT,
                              sign,
                              select_value(builder,
                                           SPIRV_UINT,
                                           infinity,
                                           constant(builder, 0x7F800000U),
                                           constant(builder, 0x7F7FFFFFU))),
                       value);
    return instruction->float_ftz ? select_value(builder, SPIRV_UINT, flush, sign, value) : value;
}

static uint32_t
half_to_float(spirv_builder_t *builder, uint32_t value, const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    uint32_t       sign = binary(
        builder,
        SPIRV_SHIFT_LEFT,
        SPIRV_UINT,
        binary(builder,
               SPIRV_AND,
               SPIRV_UINT,
               binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, 15)),
               constant(builder, 1)),
        constant(builder, 31));
    if (instruction->sources[0].absolute)
    {
        sign = zero;
    }
    if (instruction->sources[0].negate)
    {
        sign = binary(builder, SPIRV_XOR, SPIRV_UINT, sign, constant(builder, 0x80000000U));
    }
    const uint32_t exponent
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, 10)),
                 constant(builder, 31));
    const uint32_t fraction
        = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x3FFU));
    const uint32_t distance = binary(builder,
                                     SPIRV_AND,
                                     SPIRV_UINT,
                                     binary(builder,
                                            SPIRV_SUBTRACT,
                                            SPIRV_UINT,
                                            float_leading_zeros(builder, fraction),
                                            constant(builder, 21)),
                                     constant(builder, 31));
    const uint32_t normal   = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, exponent, zero);
    const uint32_t mantissa
        = select_value(builder,
                       SPIRV_UINT,
                       normal,
                       fraction,
                       binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, fraction, distance));
    const uint32_t result_exponent = select_value(
        builder,
        SPIRV_UINT,
        normal,
        binary(builder, SPIRV_ADD, SPIRV_UINT, exponent, constant(builder, 112)),
        binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 113), distance));
    uint32_t result = binary(
        builder,
        SPIRV_OR,
        SPIRV_UINT,
        sign,
        binary(
            builder,
            SPIRV_OR,
            SPIRV_UINT,
            binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, result_exponent, constant(builder, 23)),
            binary(builder,
                   SPIRV_SHIFT_LEFT,
                   SPIRV_UINT,
                   binary(builder, SPIRV_AND, SPIRV_UINT, mantissa, constant(builder, 0x3FFU)),
                   constant(builder, 13))));
    const uint32_t tiny  = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, exponent, zero);
    const uint32_t empty = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, fraction, zero);
    result               = select_value(
        builder,
        SPIRV_UINT,
        binary(
            builder, SPIRV_BOOL_AND, SPIRV_BOOL, tiny, instruction->float_ftz ? SPIRV_TRUE : empty),
        sign,
        result);
    return select_value(
        builder,
        SPIRV_UINT,
        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, exponent, constant(builder, 31)),
        select_value(builder,
                     SPIRV_UINT,
                     empty,
                     binary(builder, SPIRV_OR, SPIRV_UINT, sign, constant(builder, 0x7F800000U)),
                     constant(builder, 0x7FFFFFFFU)),
        result);
}

static uint32_t
float_to_half(spirv_builder_t *builder, uint32_t value, const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t one  = constant(builder, 1);
    const uint32_t sign
        = binary(builder,
                 SPIRV_SHIFT_RIGHT,
                 SPIRV_UINT,
                 binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x80000000U)),
                 constant(builder, 16));
    const uint32_t source_exponent
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, 23)),
                 constant(builder, 255));
    const uint32_t normal   = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, source_exponent, zero);
    uint32_t       exponent = binary(builder,
                               SPIRV_SUBTRACT,
                               SPIRV_UINT,
                               select_value(builder, SPIRV_UINT, normal, source_exponent, one),
                               constant(builder, 112));
    const uint32_t significand
        = binary(builder,
                 SPIRV_OR,
                 SPIRV_UINT,
                 binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x007FFFFFU)),
                 select_value(builder, SPIRV_UINT, normal, constant(builder, 0x00800000U), zero));
    const uint32_t tiny = compare(builder, exponent, one, GPU_SHADER_COMPARE_LT, 1);
    const uint32_t distance
        = select_value(builder,
                       SPIRV_UINT,
                       tiny,
                       binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 14), exponent),
                       constant(builder, 13));
    const uint32_t large
        = compare(builder, distance, constant(builder, 32), GPU_SHADER_COMPARE_GE, 0);
    const uint32_t shift = binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31));
    const uint32_t mask  = binary(builder,
                                 SPIRV_SUBTRACT,
                                 SPIRV_UINT,
                                 binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, one, shift),
                                 one);
    const uint32_t discarded
        = select_value(builder,
                       SPIRV_UINT,
                       large,
                       significand,
                       binary(builder, SPIRV_AND, SPIRV_UINT, significand, mask));
    uint32_t mantissa
        = select_value(builder,
                       SPIRV_UINT,
                       large,
                       zero,
                       binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, significand, shift));
    const uint32_t negative  = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, sign, zero);
    uint32_t       increment = SPIRV_FALSE;
    if (instruction->float_rounding == GPU_SHADER_ROUND_NEAREST)
    {
        const uint32_t half_shift
            = binary(builder,
                     SPIRV_AND,
                     SPIRV_UINT,
                     binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, distance, one),
                     constant(builder, 31));
        const uint32_t half
            = select_value(builder,
                           SPIRV_UINT,
                           large,
                           constant(builder, UINT32_MAX),
                           binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, one, half_shift));
        increment = binary(builder,
                           SPIRV_BOOL_OR,
                           SPIRV_BOOL,
                           compare(builder, discarded, half, GPU_SHADER_COMPARE_GT, 0),
                           binary(builder,
                                  SPIRV_BOOL_AND,
                                  SPIRV_BOOL,
                                  binary(builder, SPIRV_EQUAL, SPIRV_BOOL, discarded, half),
                                  binary(builder,
                                         SPIRV_NOT_EQUAL,
                                         SPIRV_BOOL,
                                         binary(builder, SPIRV_AND, SPIRV_UINT, mantissa, one),
                                         zero)));
    }
    else if (instruction->float_rounding != GPU_SHADER_ROUND_ZERO)
    {
        increment = binary(builder,
                           SPIRV_BOOL_AND,
                           SPIRV_BOOL,
                           binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, discarded, zero),
                           instruction->float_rounding == GPU_SHADER_ROUND_DOWN
                               ? negative
                               : unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative));
    }
    mantissa = binary(builder,
                      SPIRV_ADD,
                      SPIRV_UINT,
                      mantissa,
                      select_value(builder, SPIRV_UINT, increment, one, zero));
    exponent = select_value(builder, SPIRV_UINT, tiny, one, exponent);
    const uint32_t carry
        = compare(builder, mantissa, constant(builder, 0x800), GPU_SHADER_COMPARE_GE, 0);
    mantissa = select_value(builder,
                            SPIRV_UINT,
                            carry,
                            binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, mantissa, one),
                            mantissa);
    exponent = binary(builder,
                      SPIRV_ADD,
                      SPIRV_UINT,
                      exponent,
                      select_value(builder, SPIRV_UINT, carry, one, zero));
    const uint32_t half_normal
        = compare(builder, mantissa, constant(builder, 0x400), GPU_SHADER_COMPARE_GE, 0);
    uint32_t result = binary(
        builder,
        SPIRV_OR,
        SPIRV_UINT,
        sign,
        binary(builder,
               SPIRV_OR,
               SPIRV_UINT,
               select_value(
                   builder,
                   SPIRV_UINT,
                   half_normal,
                   binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, exponent, constant(builder, 10)),
                   zero),
               binary(builder, SPIRV_AND, SPIRV_UINT, mantissa, constant(builder, 0x3FF))));
    if (instruction->float_ftz)
    {
        result = select_value(builder, SPIRV_UINT, half_normal, result, sign);
    }
    const uint32_t infinity = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST ? SPIRV_TRUE
                              : instruction->float_rounding == GPU_SHADER_ROUND_ZERO  ? SPIRV_FALSE
                              : instruction->float_rounding == GPU_SHADER_ROUND_DOWN
                                  ? negative
                                  : unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative);
    result
        = select_value(builder,
                       SPIRV_UINT,
                       compare(builder, exponent, constant(builder, 31), GPU_SHADER_COMPARE_GE, 1),
                       binary(builder,
                              SPIRV_OR,
                              SPIRV_UINT,
                              sign,
                              select_value(builder,
                                           SPIRV_UINT,
                                           infinity,
                                           constant(builder, 0x7C00),
                                           constant(builder, 0x7BFF))),
                       result);
    const uint32_t magnitude
        = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x7FFFFFFFU));
    result = select_value(
        builder,
        SPIRV_UINT,
        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, magnitude, constant(builder, 0x7F800000U)),
        binary(builder, SPIRV_OR, SPIRV_UINT, sign, constant(builder, 0x7C00)),
        result);
    return select_value(
        builder, SPIRV_UINT, float_nan(builder, value), constant(builder, 0x7FFF), result);
}

static uint32_t
float_round_integer(spirv_builder_t                *builder,
                    uint32_t                        value,
                    const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t one  = constant(builder, 1);
    const uint32_t sign
        = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x80000000U));
    const uint32_t magnitude
        = binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x7FFFFFFFU));
    const uint32_t exponent
        = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, magnitude, constant(builder, 23));
    const uint32_t negative = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, sign, zero);
    const uint32_t outward  = instruction->float_rounding == GPU_SHADER_ROUND_DOWN ? negative
                              : instruction->float_rounding == GPU_SHADER_ROUND_UP
                                  ? unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative)
                                  : SPIRV_FALSE;
    const uint32_t distance
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 150), exponent),
                 constant(builder, 31));
    const uint32_t unit      = binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, one, distance);
    const uint32_t mask      = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, unit, one);
    const uint32_t discarded = binary(builder, SPIRV_AND, SPIRV_UINT, value, mask);
    const uint32_t integral  = binary(
        builder, SPIRV_AND, SPIRV_UINT, value, unary(builder, SPIRV_NOT, SPIRV_UINT, mask));
    uint32_t increment;
    uint32_t small_increment;
    if (instruction->float_rounding == GPU_SHADER_ROUND_NEAREST)
    {
        const uint32_t half = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, unit, one);
        increment           = binary(builder,
                           SPIRV_BOOL_OR,
                           SPIRV_BOOL,
                           compare(builder, discarded, half, GPU_SHADER_COMPARE_GT, 0),
                           binary(builder,
                                  SPIRV_BOOL_AND,
                                  SPIRV_BOOL,
                                  binary(builder, SPIRV_EQUAL, SPIRV_BOOL, discarded, half),
                                  binary(builder,
                                         SPIRV_NOT_EQUAL,
                                         SPIRV_BOOL,
                                         binary(builder, SPIRV_AND, SPIRV_UINT, integral, unit),
                                         zero)));
        small_increment
            = compare(builder, magnitude, constant(builder, 0x3F000000U), GPU_SHADER_COMPARE_GT, 0);
    }
    else
    {
        increment       = binary(builder,
                           SPIRV_BOOL_AND,
                           SPIRV_BOOL,
                           outward,
                           binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, discarded, zero));
        small_increment = binary(builder,
                                 SPIRV_BOOL_AND,
                                 SPIRV_BOOL,
                                 outward,
                                 binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, magnitude, zero));
    }
    uint32_t       result = binary(builder,
                             SPIRV_ADD,
                             SPIRV_UINT,
                             integral,
                             select_value(builder, SPIRV_UINT, increment, unit, zero));
    const uint32_t small  = binary(
        builder,
        SPIRV_OR,
        SPIRV_UINT,
        sign,
        select_value(builder, SPIRV_UINT, small_increment, constant(builder, 0x3F800000U), zero));
    result
        = select_value(builder,
                       SPIRV_UINT,
                       compare(builder, exponent, constant(builder, 127), GPU_SHADER_COMPARE_LT, 0),
                       small,
                       result);
    result
        = select_value(builder,
                       SPIRV_UINT,
                       compare(builder, exponent, constant(builder, 150), GPU_SHADER_COMPARE_GE, 0),
                       value,
                       result);
    return select_value(
        builder, SPIRV_UINT, float_nan(builder, value), constant(builder, 0x7FFFFFFFU), result);
}

static uint32_t
float_to_integer(spirv_builder_t                *builder,
                 uint32_t                        value,
                 const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero     = constant(builder, 0);
    const uint32_t one      = constant(builder, 1);
    const uint32_t negative = compare(builder, value, zero, GPU_SHADER_COMPARE_LT, 1);
    const uint32_t exponent
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, 23)),
                 constant(builder, 255));
    const uint32_t significand
        = binary(builder,
                 SPIRV_OR,
                 SPIRV_UINT,
                 binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x007FFFFFU)),
                 select_value(builder,
                              SPIRV_UINT,
                              binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, exponent, zero),
                              constant(builder, 0x00800000U),
                              zero));
    const uint32_t distance
        = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 150), exponent);
    const uint32_t shift = binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31));
    const uint32_t tiny
        = compare(builder, distance, constant(builder, 32), GPU_SHADER_COMPARE_GE, 0);
    const uint32_t mask = binary(builder,
                                 SPIRV_SUBTRACT,
                                 SPIRV_UINT,
                                 binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, one, shift),
                                 one);
    const uint32_t discarded
        = select_value(builder,
                       SPIRV_UINT,
                       tiny,
                       significand,
                       binary(builder, SPIRV_AND, SPIRV_UINT, significand, mask));
    uint32_t magnitude
        = select_value(builder,
                       SPIRV_UINT,
                       tiny,
                       zero,
                       binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, significand, shift));
    uint32_t increment = SPIRV_FALSE;
    if (instruction->float_rounding == GPU_SHADER_ROUND_NEAREST)
    {
        const uint32_t half_shift
            = binary(builder,
                     SPIRV_AND,
                     SPIRV_UINT,
                     binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, distance, one),
                     constant(builder, 31));
        const uint32_t half
            = select_value(builder,
                           SPIRV_UINT,
                           tiny,
                           constant(builder, UINT32_MAX),
                           binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, one, half_shift));
        increment = binary(builder,
                           SPIRV_BOOL_OR,
                           SPIRV_BOOL,
                           compare(builder, discarded, half, GPU_SHADER_COMPARE_GT, 0),
                           binary(builder,
                                  SPIRV_BOOL_AND,
                                  SPIRV_BOOL,
                                  binary(builder, SPIRV_EQUAL, SPIRV_BOOL, discarded, half),
                                  binary(builder,
                                         SPIRV_NOT_EQUAL,
                                         SPIRV_BOOL,
                                         binary(builder, SPIRV_AND, SPIRV_UINT, magnitude, one),
                                         zero)));
    }
    else if (instruction->float_rounding != GPU_SHADER_ROUND_ZERO)
    {
        increment = binary(builder,
                           SPIRV_BOOL_AND,
                           SPIRV_BOOL,
                           binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, discarded, zero),
                           instruction->float_rounding == GPU_SHADER_ROUND_DOWN
                               ? negative
                               : unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, negative));
    }
    magnitude = binary(builder,
                       SPIRV_ADD,
                       SPIRV_UINT,
                       magnitude,
                       select_value(builder, SPIRV_UINT, increment, one, zero));
    const uint32_t integral_shift
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, exponent, constant(builder, 150)),
                 constant(builder, 31));
    magnitude
        = select_value(builder,
                       SPIRV_UINT,
                       compare(builder, exponent, constant(builder, 150), GPU_SHADER_COMPARE_GE, 0),
                       binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, significand, integral_shift),
                       magnitude);
    const uint32_t limit = select_value(
        builder,
        SPIRV_UINT,
        negative,
        instruction->conversion_signed ? constant(builder, 0x80000000U) : zero,
        constant(builder, instruction->conversion_signed ? 0x7FFFFFFFU : UINT32_MAX));
    magnitude = select_value(
        builder,
        SPIRV_UINT,
        binary(builder,
               SPIRV_BOOL_OR,
               SPIRV_BOOL,
               compare(builder, exponent, constant(builder, 159), GPU_SHADER_COMPARE_GE, 0),
               compare(builder, magnitude, limit, GPU_SHADER_COMPARE_GT, 0)),
        limit,
        magnitude);
    const uint32_t result
        = select_value(builder,
                       SPIRV_UINT,
                       negative,
                       binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, zero, magnitude),
                       magnitude);
    return select_value(builder,
                        SPIRV_UINT,
                        float_nan(builder, value),
                        instruction->float_nan_to_zero ? zero : constant(builder, 0x80000000U),
                        result);
}

static uint32_t
integer_to_float(spirv_builder_t                *builder,
                 uint32_t                        value,
                 const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t sign
        = instruction->conversion_signed
              ? binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x80000000U))
              : zero;
    const uint32_t magnitude
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, sign, zero),
                       binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, zero, value),
                       value);
    const uint32_t leading = binary(builder,
                                    SPIRV_SUBTRACT,
                                    SPIRV_UINT,
                                    constant(builder, 31),
                                    float_leading_zeros(builder, magnitude));
    const uint32_t left_shift
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 26), leading),
                 constant(builder, 31));
    const uint32_t significand = select_value(
        builder,
        SPIRV_UINT,
        compare(builder, leading, constant(builder, 26), GPU_SHADER_COMPARE_GT, 0),
        float_shift_jam(
            builder,
            magnitude,
            binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, leading, constant(builder, 26))),
        binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, magnitude, left_shift));
    const uint32_t exponent
        = binary(builder, SPIRV_ADD, SPIRV_UINT, constant(builder, 127), leading);
    return select_value(builder,
                        SPIRV_UINT,
                        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, magnitude, zero),
                        zero,
                        float_round(builder, significand, exponent, sign, instruction));
}

static uint32_t
float_add(spirv_builder_t                *builder,
          uint32_t                        first,
          uint32_t                        second,
          const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero     = constant(builder, 0);
    const uint32_t sign_bit = constant(builder, 0x80000000U);
    const uint32_t first_magnitude
        = binary(builder, SPIRV_AND, SPIRV_UINT, first, constant(builder, 0x7FFFFFFFU));
    const uint32_t second_magnitude
        = binary(builder, SPIRV_AND, SPIRV_UINT, second, constant(builder, 0x7FFFFFFFU));
    const uint32_t swap
        = compare(builder, first_magnitude, second_magnitude, GPU_SHADER_COMPARE_LT, 0);
    const uint32_t larger  = select_value(builder, SPIRV_UINT, swap, second, first);
    const uint32_t smaller = select_value(builder, SPIRV_UINT, swap, first, second);
    const uint32_t sign    = binary(builder, SPIRV_AND, SPIRV_UINT, larger, sign_bit);
    const uint32_t matching_signs
        = binary(builder,
                 SPIRV_EQUAL,
                 SPIRV_BOOL,
                 binary(builder,
                        SPIRV_AND,
                        SPIRV_UINT,
                        binary(builder, SPIRV_XOR, SPIRV_UINT, first, second),
                        sign_bit),
                 zero);
    uint32_t       exponent, smaller_exponent;
    const uint32_t large_mantissa = binary(builder,
                                           SPIRV_SHIFT_LEFT,
                                           SPIRV_UINT,
                                           float_mantissa(builder, larger, &exponent),
                                           constant(builder, 3));
    const uint32_t small_mantissa = binary(builder,
                                           SPIRV_SHIFT_LEFT,
                                           SPIRV_UINT,
                                           float_mantissa(builder, smaller, &smaller_exponent),
                                           constant(builder, 3));
    const uint32_t aligned
        = float_shift_jam(builder,
                          small_mantissa,
                          binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, exponent, smaller_exponent));
    uint32_t significand
        = select_value(builder,
                       SPIRV_UINT,
                       matching_signs,
                       binary(builder, SPIRV_ADD, SPIRV_UINT, large_mantissa, aligned),
                       binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, large_mantissa, aligned));
    const uint32_t empty = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, significand, zero);
    const uint32_t carry
        = compare(builder, significand, constant(builder, 0x08000000U), GPU_SHADER_COMPARE_GE, 0);
    significand = select_value(builder,
                               SPIRV_UINT,
                               carry,
                               float_shift_jam(builder, significand, constant(builder, 1)),
                               significand);
    exponent    = binary(builder,
                      SPIRV_ADD,
                      SPIRV_UINT,
                      exponent,
                      select_value(builder, SPIRV_UINT, carry, constant(builder, 1), zero));
    const uint32_t available
        = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, exponent, constant(builder, 1));
    const uint32_t leading = binary(builder,
                                    SPIRV_SUBTRACT,
                                    SPIRV_UINT,
                                    float_leading_zeros(builder, significand),
                                    constant(builder, 5));
    const uint32_t distance
        = select_value(builder,
                       SPIRV_UINT,
                       compare(builder, leading, available, GPU_SHADER_COMPARE_LT, 0),
                       leading,
                       available);
    significand    = binary(builder,
                         SPIRV_SHIFT_LEFT,
                         SPIRV_UINT,
                         significand,
                         binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31)));
    exponent       = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, exponent, distance);
    uint32_t value = float_round(builder, significand, exponent, sign, instruction);
    value          = select_value(
        builder,
        SPIRV_UINT,
        empty,
        select_value(builder,
                     SPIRV_UINT,
                     matching_signs,
                     sign,
                     instruction->float_rounding == GPU_SHADER_ROUND_DOWN ? sign_bit : zero),
        value);
    const uint32_t first_infinite
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, first_magnitude, constant(builder, 0x7F800000U));
    const uint32_t second_infinite = binary(
        builder, SPIRV_EQUAL, SPIRV_BOOL, second_magnitude, constant(builder, 0x7F800000U));
    const uint32_t invalid_infinity
        = binary(builder,
                 SPIRV_BOOL_AND,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_BOOL_AND, SPIRV_BOOL, first_infinite, second_infinite),
                 unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, matching_signs));
    value
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, first_infinite, second_infinite),
                       larger,
                       value);
    return select_value(builder,
                        SPIRV_UINT,
                        binary(builder,
                               SPIRV_BOOL_OR,
                               SPIRV_BOOL,
                               invalid_infinity,
                               binary(builder,
                                      SPIRV_BOOL_OR,
                                      SPIRV_BOOL,
                                      float_nan(builder, first),
                                      float_nan(builder, second))),
                        constant(builder, 0x7FFFFFFFU),
                        value);
}

static uint32_t
float_multiply(spirv_builder_t                *builder,
               uint32_t                        first,
               uint32_t                        second,
               const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t sign = binary(builder,
                                 SPIRV_AND,
                                 SPIRV_UINT,
                                 binary(builder, SPIRV_XOR, SPIRV_UINT, first, second),
                                 constant(builder, 0x80000000U));
    const uint32_t left
        = binary(builder, SPIRV_AND, SPIRV_UINT, first, constant(builder, 0x7FFFFFFFU));
    const uint32_t right
        = binary(builder, SPIRV_AND, SPIRV_UINT, second, constant(builder, 0x7FFFFFFFU));
    uint32_t       left_exponent, right_exponent;
    uint32_t       left_mantissa  = float_mantissa(builder, first, &left_exponent);
    uint32_t       right_mantissa = float_mantissa(builder, second, &right_exponent);
    const uint32_t left_shift     = binary(builder,
                                       SPIRV_SUBTRACT,
                                       SPIRV_UINT,
                                       float_leading_zeros(builder, left_mantissa),
                                       constant(builder, 8));
    const uint32_t right_shift    = binary(builder,
                                        SPIRV_SUBTRACT,
                                        SPIRV_UINT,
                                        float_leading_zeros(builder, right_mantissa),
                                        constant(builder, 8));
    left_mantissa  = binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, left_mantissa, left_shift);
    right_mantissa = binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, right_mantissa, right_shift);
    left_exponent  = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, left_exponent, left_shift);
    right_exponent = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, right_exponent, right_shift);
    const uint32_t product
        = binary(builder, SPIRV_MULTIPLY_UNSIGNED_WIDE, SPIRV_PAIR, left_mantissa, right_mantissa);
    const uint32_t lower = extract(builder, product, 0);
    const uint32_t upper = extract(builder, product, 1);
    const uint32_t high
        = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, upper, constant(builder, 15));
    const uint32_t distance = binary(builder, SPIRV_ADD, SPIRV_UINT, constant(builder, 20), high);
    const uint32_t mask
        = binary(builder,
                 SPIRV_SUBTRACT,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, constant(builder, 1), distance),
                 constant(builder, 1));
    const uint32_t significand = binary(
        builder,
        SPIRV_OR,
        SPIRV_UINT,
        binary(
            builder,
            SPIRV_OR,
            SPIRV_UINT,
            binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, lower, distance),
            binary(builder,
                   SPIRV_SHIFT_LEFT,
                   SPIRV_UINT,
                   upper,
                   binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 32), distance))),
        select_value(builder,
                     SPIRV_UINT,
                     binary(builder,
                            SPIRV_NOT_EQUAL,
                            SPIRV_BOOL,
                            binary(builder, SPIRV_AND, SPIRV_UINT, lower, mask),
                            zero),
                     constant(builder, 1),
                     zero));
    const uint32_t exponent
        = binary(builder,
                 SPIRV_SUBTRACT,
                 SPIRV_UINT,
                 binary(builder,
                        SPIRV_ADD,
                        SPIRV_UINT,
                        binary(builder, SPIRV_ADD, SPIRV_UINT, left_exponent, right_exponent),
                        high),
                 constant(builder, 127));
    uint32_t       value      = float_round(builder, significand, exponent, sign, instruction);
    const uint32_t left_zero  = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, left, zero);
    const uint32_t right_zero = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, right, zero);
    const uint32_t left_infinite
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, left, constant(builder, 0x7F800000U));
    const uint32_t right_infinite
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, right, constant(builder, 0x7F800000U));
    value = select_value(builder,
                         SPIRV_UINT,
                         binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, left_zero, right_zero),
                         sign,
                         value);
    value
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, left_infinite, right_infinite),
                       binary(builder, SPIRV_OR, SPIRV_UINT, sign, constant(builder, 0x7F800000U)),
                       value);
    const uint32_t invalid
        = binary(builder,
                 SPIRV_BOOL_OR,
                 SPIRV_BOOL,
                 binary(builder,
                        SPIRV_BOOL_OR,
                        SPIRV_BOOL,
                        float_nan(builder, first),
                        float_nan(builder, second)),
                 binary(builder,
                        SPIRV_BOOL_OR,
                        SPIRV_BOOL,
                        binary(builder, SPIRV_BOOL_AND, SPIRV_BOOL, left_infinite, right_zero),
                        binary(builder, SPIRV_BOOL_AND, SPIRV_BOOL, right_infinite, left_zero)));
    return select_value(builder, SPIRV_UINT, invalid, constant(builder, 0x7FFFFFFFU), value);
}

static uint32_t
wide_less(spirv_builder_t *builder, spirv_wide_t first, spirv_wide_t second)
{
    return binary(builder,
                  SPIRV_BOOL_OR,
                  SPIRV_BOOL,
                  compare(builder, first.high, second.high, GPU_SHADER_COMPARE_LT, 0),
                  binary(builder,
                         SPIRV_BOOL_AND,
                         SPIRV_BOOL,
                         binary(builder, SPIRV_EQUAL, SPIRV_BOOL, first.high, second.high),
                         compare(builder, first.low, second.low, GPU_SHADER_COMPARE_LT, 0)));
}

static spirv_wide_t
wide_select(spirv_builder_t *builder, uint32_t condition, spirv_wide_t first, spirv_wide_t second)
{
    return (spirv_wide_t) { select_value(builder, SPIRV_UINT, condition, first.low, second.low),
                            select_value(builder, SPIRV_UINT, condition, first.high, second.high) };
}

static spirv_wide_t
wide_shift_left(spirv_builder_t *builder, spirv_wide_t value, uint32_t distance)
{
    const uint32_t zero   = constant(builder, 0);
    const uint32_t amount = binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31));
    const uint32_t reverse
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 32), amount),
                 constant(builder, 31));
    const uint32_t cross
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder, SPIRV_EQUAL, SPIRV_BOOL, amount, zero),
                       zero,
                       binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value.low, reverse));
    const uint32_t lower = binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, value.low, amount);
    const uint32_t upper
        = binary(builder,
                 SPIRV_OR,
                 SPIRV_UINT,
                 cross,
                 binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, value.high, amount));
    return wide_select(builder,
                       compare(builder, distance, constant(builder, 32), GPU_SHADER_COMPARE_GE, 0),
                       (spirv_wide_t) { zero, lower },
                       (spirv_wide_t) { lower, upper });
}

static spirv_wide_t
wide_shift_jam(spirv_builder_t *builder, spirv_wide_t value, uint32_t distance)
{
    const uint32_t zero   = constant(builder, 0);
    const uint32_t one    = constant(builder, 1);
    const uint32_t amount = binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31));
    const uint32_t reverse
        = binary(builder,
                 SPIRV_AND,
                 SPIRV_UINT,
                 binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 32), amount),
                 constant(builder, 31));
    const uint32_t mask       = binary(builder,
                                 SPIRV_SUBTRACT,
                                 SPIRV_UINT,
                                 binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, one, amount),
                                 one);
    const uint32_t lower_lost = binary(builder,
                                       SPIRV_NOT_EQUAL,
                                       SPIRV_BOOL,
                                       binary(builder, SPIRV_AND, SPIRV_UINT, value.low, mask),
                                       zero);
    const uint32_t upper_lost
        = binary(builder,
                 SPIRV_BOOL_OR,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, value.low, zero),
                 binary(builder,
                        SPIRV_NOT_EQUAL,
                        SPIRV_BOOL,
                        binary(builder, SPIRV_AND, SPIRV_UINT, value.high, mask),
                        zero));
    const uint32_t cross
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder, SPIRV_EQUAL, SPIRV_BOOL, amount, zero),
                       zero,
                       binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, value.high, reverse));
    const uint32_t upper = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value.high, amount);
    const uint32_t lower
        = binary(builder,
                 SPIRV_OR,
                 SPIRV_UINT,
                 binary(builder,
                        SPIRV_OR,
                        SPIRV_UINT,
                        cross,
                        binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value.low, amount)),
                 select_value(builder, SPIRV_UINT, lower_lost, one, zero));
    const uint32_t     only_upper = binary(builder,
                                       SPIRV_OR,
                                       SPIRV_UINT,
                                       upper,
                                       select_value(builder, SPIRV_UINT, upper_lost, one, zero));
    const spirv_wide_t shifted
        = wide_select(builder,
                      compare(builder, distance, constant(builder, 32), GPU_SHADER_COMPARE_GE, 0),
                      (spirv_wide_t) { only_upper, zero },
                      (spirv_wide_t) { lower, upper });
    const uint32_t any = binary(builder,
                                SPIRV_NOT_EQUAL,
                                SPIRV_BOOL,
                                binary(builder, SPIRV_OR, SPIRV_UINT, value.low, value.high),
                                zero);
    return wide_select(builder,
                       compare(builder, distance, constant(builder, 64), GPU_SHADER_COMPARE_GE, 0),
                       (spirv_wide_t) { select_value(builder, SPIRV_UINT, any, one, zero), zero },
                       shifted);
}

static uint32_t
float_normalize(spirv_builder_t *builder, uint32_t value, uint32_t *exponent)
{
    uint32_t       mantissa = float_mantissa(builder, value, exponent);
    const uint32_t distance = binary(builder,
                                     SPIRV_SUBTRACT,
                                     SPIRV_UINT,
                                     float_leading_zeros(builder, mantissa),
                                     constant(builder, 8));
    mantissa                = binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, mantissa, distance);
    *exponent               = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, *exponent, distance);
    return mantissa;
}

static uint32_t
float_fused(spirv_builder_t                *builder,
            uint32_t                        first,
            uint32_t                        second,
            uint32_t                        third,
            const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero         = constant(builder, 0);
    const uint32_t one          = constant(builder, 1);
    const uint32_t sign_bit     = constant(builder, 0x80000000U);
    const uint32_t product_sign = binary(builder,
                                         SPIRV_AND,
                                         SPIRV_UINT,
                                         binary(builder, SPIRV_XOR, SPIRV_UINT, first, second),
                                         sign_bit);
    const uint32_t addend_sign  = binary(builder, SPIRV_AND, SPIRV_UINT, third, sign_bit);
    const uint32_t matching_signs
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, product_sign, addend_sign);
    uint32_t       left_exponent, right_exponent, addend_exponent;
    const uint32_t left_mantissa   = float_normalize(builder, first, &left_exponent);
    const uint32_t right_mantissa  = float_normalize(builder, second, &right_exponent);
    const uint32_t addend_mantissa = float_normalize(builder, third, &addend_exponent);
    const uint32_t multiplied
        = binary(builder, SPIRV_MULTIPLY_UNSIGNED_WIDE, SPIRV_PAIR, left_mantissa, right_mantissa);
    spirv_wide_t   product = { extract(builder, multiplied, 0), extract(builder, multiplied, 1) };
    const uint32_t high
        = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, product.high, constant(builder, 15));
    product = wide_shift_left(
        builder, product, binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 16), high));
    const spirv_wide_t addend
        = wide_shift_left(builder, (spirv_wide_t) { addend_mantissa, zero }, constant(builder, 39));
    const uint32_t product_exponent
        = binary(builder,
                 SPIRV_SUBTRACT,
                 SPIRV_UINT,
                 binary(builder,
                        SPIRV_ADD,
                        SPIRV_UINT,
                        binary(builder, SPIRV_ADD, SPIRV_UINT, left_exponent, right_exponent),
                        high),
                 constant(builder, 127));
    const uint32_t product_zero
        = binary(builder,
                 SPIRV_EQUAL,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_OR, SPIRV_UINT, product.low, product.high),
                 zero);
    const uint32_t addend_zero   = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, addend_mantissa, zero);
    const uint32_t addend_larger = binary(
        builder,
        SPIRV_BOOL_OR,
        SPIRV_BOOL,
        product_zero,
        binary(
            builder,
            SPIRV_BOOL_AND,
            SPIRV_BOOL,
            unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, addend_zero),
            binary(
                builder,
                SPIRV_BOOL_OR,
                SPIRV_BOOL,
                compare(builder, product_exponent, addend_exponent, GPU_SHADER_COMPARE_LT, 1),
                binary(builder,
                       SPIRV_BOOL_AND,
                       SPIRV_BOOL,
                       binary(builder, SPIRV_EQUAL, SPIRV_BOOL, product_exponent, addend_exponent),
                       wide_less(builder, product, addend)))));
    const spirv_wide_t larger  = wide_select(builder, addend_larger, addend, product);
    spirv_wide_t       smaller = wide_select(builder, addend_larger, product, addend);
    uint32_t           exponent
        = select_value(builder, SPIRV_UINT, addend_larger, addend_exponent, product_exponent);
    const uint32_t smaller_exponent
        = select_value(builder, SPIRV_UINT, addend_larger, product_exponent, addend_exponent);
    const uint32_t sign
        = select_value(builder, SPIRV_UINT, addend_larger, addend_sign, product_sign);
    smaller = wide_shift_jam(
        builder, smaller, binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, exponent, smaller_exponent));
    const uint32_t sum = binary(builder, SPIRV_ADD_CARRY, SPIRV_PAIR, larger.low, smaller.low);
    const uint32_t difference
        = binary(builder, SPIRV_SUBTRACT_BORROW, SPIRV_PAIR, larger.low, smaller.low);
    const spirv_wide_t added
        = { extract(builder, sum, 0),
            binary(builder,
                   SPIRV_ADD,
                   SPIRV_UINT,
                   binary(builder, SPIRV_ADD, SPIRV_UINT, larger.high, smaller.high),
                   extract(builder, sum, 1)) };
    const spirv_wide_t subtracted
        = { extract(builder, difference, 0),
            binary(builder,
                   SPIRV_SUBTRACT,
                   SPIRV_UINT,
                   binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, larger.high, smaller.high),
                   extract(builder, difference, 1)) };
    spirv_wide_t   significand = wide_select(builder, matching_signs, added, subtracted);
    const uint32_t empty
        = binary(builder,
                 SPIRV_EQUAL,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_OR, SPIRV_UINT, significand.low, significand.high),
                 zero);
    const uint32_t carry
        = binary(builder,
                 SPIRV_NOT_EQUAL,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_AND, SPIRV_UINT, significand.high, sign_bit),
                 zero);
    significand
        = wide_select(builder, carry, wide_shift_jam(builder, significand, one), significand);
    exponent = binary(builder,
                      SPIRV_ADD,
                      SPIRV_UINT,
                      exponent,
                      select_value(builder, SPIRV_UINT, carry, one, zero));
    const uint32_t leading
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, significand.high, zero),
                       float_leading_zeros(builder, significand.high),
                       binary(builder,
                              SPIRV_ADD,
                              SPIRV_UINT,
                              constant(builder, 32),
                              float_leading_zeros(builder, significand.low)));
    const uint32_t distance    = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, leading, one);
    significand                = wide_shift_left(builder, significand, distance);
    exponent                   = binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, exponent, distance);
    const spirv_wide_t reduced = wide_shift_jam(builder, significand, constant(builder, 36));
    uint32_t           value   = float_round(builder, reduced.low, exponent, sign, instruction);
    value                      = select_value(
        builder,
        SPIRV_UINT,
        empty,
        select_value(builder,
                     SPIRV_UINT,
                     matching_signs,
                     sign,
                     instruction->float_rounding == GPU_SHADER_ROUND_DOWN ? sign_bit : zero),
        value);
    const uint32_t left
        = binary(builder, SPIRV_AND, SPIRV_UINT, first, constant(builder, 0x7FFFFFFFU));
    const uint32_t right
        = binary(builder, SPIRV_AND, SPIRV_UINT, second, constant(builder, 0x7FFFFFFFU));
    const uint32_t last
        = binary(builder, SPIRV_AND, SPIRV_UINT, third, constant(builder, 0x7FFFFFFFU));
    const uint32_t left_infinite
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, left, constant(builder, 0x7F800000U));
    const uint32_t right_infinite
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, right, constant(builder, 0x7F800000U));
    const uint32_t last_infinite
        = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, last, constant(builder, 0x7F800000U));
    const uint32_t product_infinite
        = binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, left_infinite, right_infinite);
    const uint32_t invalid_product
        = binary(builder,
                 SPIRV_BOOL_OR,
                 SPIRV_BOOL,
                 binary(builder,
                        SPIRV_BOOL_AND,
                        SPIRV_BOOL,
                        left_infinite,
                        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, right, zero)),
                 binary(builder,
                        SPIRV_BOOL_AND,
                        SPIRV_BOOL,
                        right_infinite,
                        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, left, zero)));
    const uint32_t invalid_sum
        = binary(builder,
                 SPIRV_BOOL_AND,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_BOOL_AND, SPIRV_BOOL, product_infinite, last_infinite),
                 unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, matching_signs));
    const uint32_t invalid
        = binary(builder,
                 SPIRV_BOOL_OR,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, invalid_product, invalid_sum),
                 binary(builder,
                        SPIRV_BOOL_OR,
                        SPIRV_BOOL,
                        float_nan(builder, third),
                        binary(builder,
                               SPIRV_BOOL_OR,
                               SPIRV_BOOL,
                               float_nan(builder, first),
                               float_nan(builder, second))));
    value = select_value(builder, SPIRV_UINT, last_infinite, third, value);
    value = select_value(
        builder,
        SPIRV_UINT,
        product_infinite,
        binary(builder, SPIRV_OR, SPIRV_UINT, product_sign, constant(builder, 0x7F800000U)),
        value);
    return select_value(builder, SPIRV_UINT, invalid, constant(builder, 0x7FFFFFFFU), value);
}

static uint32_t
float_arithmetic(spirv_builder_t                *builder,
                 uint32_t                        first,
                 uint32_t                        second,
                 uint32_t                        third,
                 const gpu_shader_instruction_t *instruction)
{
    uint32_t value = instruction->operation == GPU_SHADER_FFMA
                         ? float_fused(builder, first, second, third, instruction)
                     : instruction->operation == GPU_SHADER_FADD
                         ? float_add(builder, first, second, instruction)
                         : float_multiply(builder, first, second, instruction);
    if (instruction->float_saturate)
    {
        const uint32_t negative
            = binary(builder,
                     SPIRV_NOT_EQUAL,
                     SPIRV_BOOL,
                     binary(builder, SPIRV_AND, SPIRV_UINT, value, constant(builder, 0x80000000U)),
                     constant(builder, 0));
        value = select_value(
            builder,
            SPIRV_UINT,
            binary(builder, SPIRV_BOOL_OR, SPIRV_BOOL, negative, float_nan(builder, value)),
            constant(builder, 0),
            select_value(
                builder,
                SPIRV_UINT,
                compare(builder, value, constant(builder, 0x3F800000U), GPU_SHADER_COMPARE_GT, 0),
                constant(builder, 0x3F800000U),
                value));
    }
    return value;
}

static uint32_t
logic_lookup(
    spirv_builder_t *builder, uint32_t first, uint32_t second, uint32_t third, uint32_t lookup)
{
    const uint32_t  original[3] = { first, second, third };
    uint32_t        inverse[3];
    const uint32_t *operand  = original;
    uint32_t       *inverted = inverse;
    for (uint32_t index = 0; index < 3U; ++index, ++operand, ++inverted)
    {
        *inverted = unary(builder, SPIRV_NOT, SPIRV_UINT, *operand);
    }
    uint32_t value = constant(builder, 0);
    for (uint32_t index = 0; index < 8U; ++index)
    {
        if (0 != (lookup & (1U << index)))
        {
            const uint32_t low  = binary(builder,
                                        SPIRV_AND,
                                        SPIRV_UINT,
                                        index & 4U ? original[0] : inverse[0],
                                        index & 2U ? original[1] : inverse[1]);
            const uint32_t term = binary(
                builder, SPIRV_AND, SPIRV_UINT, low, index & 1U ? original[2] : inverse[2]);
            value = binary(builder, SPIRV_OR, SPIRV_UINT, value, term);
        }
    }
    return value;
}

static uint32_t
modified_source(spirv_builder_t *builder, const gpu_shader_source_t *operand, uint32_t bitwise)
{
    const uint32_t value = source(builder, operand);
    if (0 == operand->negate)
    {
        return value;
    }
    return bitwise ? unary(builder, SPIRV_NOT, SPIRV_UINT, value)
                   : binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 0), value);
}

static uint32_t
predicate_word(spirv_builder_t *builder, uint32_t value)
{
    return select_value(builder, SPIRV_UINT, value, constant(builder, 1), constant(builder, 0));
}

static uint32_t
add_word(spirv_builder_t *builder, uint32_t first, uint32_t second, uint32_t *upper)
{
    const uint32_t sum = binary(builder, SPIRV_ADD_CARRY, SPIRV_PAIR, first, second);
    *upper             = binary(builder, SPIRV_ADD, SPIRV_UINT, *upper, extract(builder, sum, 1));
    return extract(builder, sum, 0);
}

static uint32_t
shift_distance(spirv_builder_t *builder, uint32_t amount, uint32_t width, uint32_t wrap)
{
    if (wrap)
    {
        return binary(builder, SPIRV_AND, SPIRV_UINT, amount, constant(builder, width - 1U));
    }
    const uint32_t limit = constant(builder, width);
    return select_value(builder,
                        SPIRV_UINT,
                        binary(builder, SPIRV_UNSIGNED_LESS, SPIRV_BOOL, amount, limit),
                        amount,
                        limit);
}

static uint32_t
funnel_word(spirv_builder_t *builder,
            uint32_t         low,
            uint32_t         high,
            uint32_t         distance,
            uint32_t         right,
            uint32_t         signed_value,
            uint32_t         output_high)
{
    const uint32_t zero = constant(builder, 0);
    const uint32_t small
        = binary(builder, SPIRV_UNSIGNED_LESS, SPIRV_BOOL, distance, constant(builder, 32));
    const uint32_t full = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, distance, constant(builder, 64));
    const uint32_t amount = binary(builder, SPIRV_AND, SPIRV_UINT, distance, constant(builder, 31));
    const uint32_t reverse = binary(builder,
                                    SPIRV_AND,
                                    SPIRV_UINT,
                                    binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, zero, amount),
                                    constant(builder, 31));
    const uint32_t empty   = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, amount, zero);
    const uint32_t fill
        = right && signed_value
              ? binary(builder, SPIRV_SHIFT_RIGHT_SIGNED, SPIRV_UINT, high, constant(builder, 31))
              : zero;
    const uint32_t right_opcode = signed_value ? SPIRV_SHIFT_RIGHT_SIGNED : SPIRV_SHIFT_RIGHT;
    uint32_t       narrow;
    uint32_t       wide;
    if (right)
    {
        if (output_high)
        {
            narrow = binary(builder, right_opcode, SPIRV_UINT, high, amount);
            wide   = fill;
        }
        else
        {
            const uint32_t cross
                = select_value(builder,
                               SPIRV_UINT,
                               empty,
                               zero,
                               binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, high, reverse));
            narrow = binary(builder,
                            SPIRV_OR,
                            SPIRV_UINT,
                            cross,
                            binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, low, amount));
            wide   = select_value(builder,
                                SPIRV_UINT,
                                full,
                                fill,
                                binary(builder, right_opcode, SPIRV_UINT, high, amount));
        }
    }
    else if (output_high)
    {
        const uint32_t cross
            = select_value(builder,
                           SPIRV_UINT,
                           empty,
                           zero,
                           binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, low, reverse));
        narrow = binary(builder,
                        SPIRV_OR,
                        SPIRV_UINT,
                        cross,
                        binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, high, amount));
        wide   = select_value(builder,
                            SPIRV_UINT,
                            full,
                            zero,
                            binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, low, amount));
    }
    else
    {
        narrow = binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, low, amount);
        wide   = zero;
    }
    return select_value(builder, SPIRV_UINT, small, narrow, wide);
}

static uint32_t
bit_field(spirv_builder_t *builder,
          uint32_t         value,
          uint32_t         offset,
          uint32_t         width,
          uint32_t         signed_value)
{
    const uint32_t result = builder->next_id++;
    EMIT(builder,
         signed_value ? SPIRV_FIELD_SIGNED : SPIRV_FIELD_UNSIGNED,
         SPIRV_UINT,
         result,
         value,
         offset,
         width);
    return result;
}

static uint32_t
leading_bit(spirv_builder_t *builder, uint32_t value, const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero = constant(builder, 0);
    if (instruction->compare_signed)
    {
        value = select_value(builder,
                             SPIRV_UINT,
                             compare(builder, value, zero, GPU_SHADER_COMPARE_LT, 1),
                             unary(builder, SPIRV_NOT, SPIRV_UINT, value),
                             value);
    }
    const uint32_t empty = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, value, zero);
    for (uint32_t distance = 1U; distance < 32U; distance *= 2U)
    {
        value = binary(
            builder,
            SPIRV_OR,
            SPIRV_UINT,
            value,
            binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, value, constant(builder, distance)));
    }
    const uint32_t bits = unary(builder, SPIRV_COUNT, SPIRV_UINT, value);
    const uint32_t result
        = instruction->return_shift
              ? binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, constant(builder, 32), bits)
              : binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, bits, constant(builder, 1));
    return select_value(builder, SPIRV_UINT, empty, constant(builder, UINT32_MAX), result);
}

static uint32_t
permute_word(spirv_builder_t          *builder,
             uint32_t                  low,
             uint32_t                  selector,
             uint32_t                  high,
             gpu_shader_permute_mode_t mode)
{
    const uint32_t zero  = constant(builder, 0);
    const uint32_t start = binary(builder, SPIRV_AND, SPIRV_UINT, selector, constant(builder, 3));
    uint32_t       value = zero;
    for (uint32_t octet = 0; octet < 4U; ++octet)
    {
        const uint32_t position  = constant(builder, octet);
        uint32_t       selection = start;
        switch (mode)
        {
            case GPU_SHADER_PERMUTE_INDEX:
                selection = bit_field(
                    builder, selector, constant(builder, 4U * octet), constant(builder, 4), 0);
                break;
            case GPU_SHADER_PERMUTE_FORWARD4:
                selection = binary(builder, SPIRV_ADD, SPIRV_UINT, start, position);
                break;
            case GPU_SHADER_PERMUTE_BACKWARD4:
                selection
                    = binary(builder,
                             SPIRV_SUBTRACT,
                             SPIRV_UINT,
                             binary(builder, SPIRV_ADD, SPIRV_UINT, start, constant(builder, 8)),
                             position);
                break;
            case GPU_SHADER_PERMUTE_CLAMP_LEFT:
            case GPU_SHADER_PERMUTE_CLAMP_RIGHT: {
                const uint32_t less
                    = binary(builder, SPIRV_UNSIGNED_LESS, SPIRV_BOOL, position, start);
                selection = mode == GPU_SHADER_PERMUTE_CLAMP_LEFT
                                ? select_value(builder, SPIRV_UINT, less, start, position)
                                : select_value(builder, SPIRV_UINT, less, position, start);
                break;
            }
            case GPU_SHADER_PERMUTE_REPLICATE16:
                selection = binary(
                    builder,
                    SPIRV_ADD,
                    SPIRV_UINT,
                    binary(builder,
                           SPIRV_MULTIPLY,
                           SPIRV_UINT,
                           binary(builder, SPIRV_AND, SPIRV_UINT, start, constant(builder, 1)),
                           constant(builder, 2)),
                    constant(builder, octet & 1U));
                break;
            default:
                break;
        }
        const uint32_t from_high
            = binary(builder,
                     SPIRV_NOT_EQUAL,
                     SPIRV_BOOL,
                     binary(builder, SPIRV_AND, SPIRV_UINT, selection, constant(builder, 4)),
                     zero);
        const uint32_t offset
            = binary(builder,
                     SPIRV_MULTIPLY,
                     SPIRV_UINT,
                     binary(builder, SPIRV_AND, SPIRV_UINT, selection, constant(builder, 3)),
                     constant(builder, 8));
        uint32_t selected = bit_field(builder,
                                      select_value(builder, SPIRV_UINT, from_high, high, low),
                                      offset,
                                      constant(builder, 8),
                                      0);
        if (mode == GPU_SHADER_PERMUTE_INDEX)
        {
            const uint32_t sign_mode
                = binary(builder,
                         SPIRV_NOT_EQUAL,
                         SPIRV_BOOL,
                         binary(builder, SPIRV_AND, SPIRV_UINT, selection, constant(builder, 8)),
                         zero);
            const uint32_t sign = select_value(builder,
                                               SPIRV_UINT,
                                               binary(builder,
                                                      SPIRV_UNSIGNED_GREATER_EQUAL,
                                                      SPIRV_BOOL,
                                                      selected,
                                                      constant(builder, 128)),
                                               constant(builder, 255),
                                               zero);
            selected            = select_value(builder, SPIRV_UINT, sign_mode, sign, selected);
        }
        value = binary(
            builder,
            SPIRV_OR,
            SPIRV_UINT,
            value,
            binary(builder, SPIRV_SHIFT_LEFT, SPIRV_UINT, selected, constant(builder, octet * 8U)));
    }
    return value;
}

static uint32_t
dot_word(spirv_builder_t                *builder,
         uint32_t                        first,
         uint32_t                        second,
         uint32_t                        accumulator,
         const gpu_shader_instruction_t *instruction)
{
    const uint32_t two   = instruction->operation == GPU_SHADER_IDP2;
    const uint32_t width = two ? 16U : 8U;
    const uint32_t count = two ? 2U : 4U;
    uint32_t       value = accumulator;
    for (uint32_t element = 0; element < count; ++element)
    {
        const uint32_t left = bit_field(builder,
                                        first,
                                        constant(builder, width * element),
                                        constant(builder, width),
                                        instruction->dot_signed[0]);
        const uint32_t right
            = bit_field(builder,
                        second,
                        constant(builder, 8U * (element + 2U * instruction->dot_high)),
                        constant(builder, 8),
                        instruction->dot_signed[1]);
        value = binary(builder,
                       SPIRV_ADD,
                       SPIRV_UINT,
                       value,
                       binary(builder, SPIRV_MULTIPLY, SPIRV_UINT, left, right));
    }
    return value;
}

static error_t
validate(const gpu_shader_program_t *program, uint32_t bounded)
{
    if (NULL == program || NULL == program->nodes || NULL == program->blocks
        || 0 == program->instruction_count || 0 == program->block_count
        || program->block_count > program->instruction_count
        || program->instruction_count > SIZE_MAX / GPU_SHADER_INSTRUCTION_SIZE
        || program->byte_size != program->instruction_count * GPU_SHADER_INSTRUCTION_SIZE)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    if (program->instruction_count > GPU_SHADER_SPIRV_INSTRUCTION_LIMIT)
    {
        return POUND_ERROR_GPU_RESOURCE_LIMIT;
    }
    const gpu_shader_node_t *node = program->nodes;
    for (size_t index = 0; index < program->instruction_count; ++index, ++node)
    {
        const gpu_shader_instruction_t *instruction = &node->instruction;
        switch (instruction->operation)
        {
            case GPU_SHADER_EXIT:
                if (!bounded
                    && (index + 1U != program->instruction_count || instruction->predicate != 7U
                        || instruction->predicate_invert))
                {
                    return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
                }
                break;
            case GPU_SHADER_BRA:
                if (!bounded)
                {
                    return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
                }
                if (node->branch_instruction >= program->instruction_count)
                {
                    return POUND_ERROR_GPU_INVALID_SHADER;
                }
                break;
            case GPU_SHADER_IADD3:
            case GPU_SHADER_IMAD:
            case GPU_SHADER_MOV:
            case GPU_SHADER_LOP3:
            case GPU_SHADER_POPC:
            case GPU_SHADER_SEL:
            case GPU_SHADER_ISETP:
            case GPU_SHADER_PLOP3:
            case GPU_SHADER_IMNMX:
            case GPU_SHADER_FSETP:
            case GPU_SHADER_FMNMX:
            case GPU_SHADER_FADD:
            case GPU_SHADER_FMUL:
            case GPU_SHADER_FFMA:
            case GPU_SHADER_F2I:
            case GPU_SHADER_I2F:
            case GPU_SHADER_FRND:
            case GPU_SHADER_I2FP:
            case GPU_SHADER_F2F:
            case GPU_SHADER_BREV:
            case GPU_SHADER_IABS:
            case GPU_SHADER_SHF:
            case GPU_SHADER_FLO:
            case GPU_SHADER_PRMT:
            case GPU_SHADER_SGXT:
            case GPU_SHADER_BMSK:
            case GPU_SHADER_LEA:
            case GPU_SHADER_IDP4:
            case GPU_SHADER_IDP2:
            case GPU_SHADER_NOP:
                break;
            default:
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
    }
    return bounded
                   || program->nodes[program->instruction_count - 1U].instruction.operation
                          == GPU_SHADER_EXIT
               ? POUND_SUCCESS
               : POUND_ERROR_GPU_UNSUPPORTED_SHADER;
}

static void
append_constant(spirv_builder_t *builder, uint32_t value)
{
    if (0 == constant(builder, value))
    {
        const uint32_t count      = builder->constant_count;
        builder->constants[count] = value;
        builder->constant_count   = count + 1U;
    }
}

static void
build_constants(spirv_builder_t *builder, const gpu_shader_program_t *program)
{
    uint32_t      *cursor = builder->constants;
    const uint32_t limit  = builder->budget ? GPU_SHADER_SPIRV_STATUS : 256U;
    for (uint32_t value = 0; value <= limit; ++value)
    {
        *cursor++ = value;
    }
    *cursor++               = UINT32_MAX;
    *cursor++               = 0x80000000U;
    builder->constant_count = limit + 3U;
    append_constant(builder, 0x7FFFFFFFU);
    append_constant(builder, 0x7F800000U);
    append_constant(builder, 0x007FFFFFU);
    append_constant(builder, 0x00800000U);
    append_constant(builder, 0x01000000U);
    append_constant(builder, 0x04000000U);
    append_constant(builder, 0x08000000U);
    append_constant(builder, 0x7F7FFFFFU);
    append_constant(builder, 0x3F800000U);
    append_constant(builder, 0x3F000000U);
    append_constant(builder, 0x3FFU);
    append_constant(builder, 0x400U);
    append_constant(builder, 0x800U);
    append_constant(builder, 0x7BFFU);
    append_constant(builder, 0x7C00U);
    append_constant(builder, 0x7FFFU);
    append_constant(builder, 0x03FFFFFEU);
    append_constant(builder, 0x03FFFFFCU);
    append_constant(builder, 0x40000000U);
    const gpu_shader_node_t *node = program->nodes;
    for (size_t index = 0; index < program->instruction_count; ++index, ++node)
    {
        const gpu_shader_source_t *operand = node->instruction.sources;
        for (uint32_t ordinal = 0; ordinal < node->instruction.source_count; ++ordinal, ++operand)
        {
            if (operand->kind == GPU_SHADER_SOURCE_IMMEDIATE)
            {
                append_constant(builder, operand->value);
            }
        }
        if (builder->budget)
        {
            append_constant(builder, (uint32_t)((index + 1U) * GPU_SHADER_INSTRUCTION_SIZE));
        }
    }
    if (builder->budget)
    {
        append_constant(builder, GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS);
        append_constant(builder, builder->budget);
    }
    builder->next_id = SPIRV_LOCAL_BOOL_POINTER + 1U + builder->constant_count;
}

static void
emit_instruction(spirv_builder_t *builder, const gpu_shader_instruction_t *instruction)
{
    const uint32_t zero    = constant(builder, 0);
    const uint32_t bitwise = instruction->operation == GPU_SHADER_POPC
                             || instruction->operation == GPU_SHADER_FLO
                             || ((instruction->operation == GPU_SHADER_IADD3
                                  || instruction->operation == GPU_SHADER_LEA)
                                 && instruction->carry_in);
    const uint32_t floating
        = instruction->operation == GPU_SHADER_FSETP || instruction->operation == GPU_SHADER_FMNMX
          || instruction->operation == GPU_SHADER_FADD || instruction->operation == GPU_SHADER_FMUL
          || instruction->operation == GPU_SHADER_F2I || instruction->operation == GPU_SHADER_FRND
          || instruction->operation == GPU_SHADER_F2F;
    const uint32_t fused = instruction->operation == GPU_SHADER_FFMA;
    const uint32_t first
        = floating || fused
              ? float_source(builder, &instruction->sources[0], instruction->float_ftz)
              : modified_source(builder, &instruction->sources[0], bitwise);
    const uint32_t second
        = floating || fused
              ? float_source(builder, &instruction->sources[1], instruction->float_ftz)
              : modified_source(builder, &instruction->sources[1], bitwise);
    const uint32_t third
        = fused ? float_source(builder, &instruction->sources[2], instruction->float_ftz)
                : modified_source(builder, &instruction->sources[2], bitwise);
    const uint32_t active
        = predicate(builder, instruction->predicate, instruction->predicate_invert);
    uint32_t value                = zero;
    uint32_t upper_result         = zero;
    uint32_t predicate_results[2] = { SPIRV_FALSE, SPIRV_FALSE };
    switch (instruction->operation)
    {
        case GPU_SHADER_MOV:
            value = first;
            break;
        case GPU_SHADER_IADD3: {
            uint32_t upper = zero;
            value          = add_word(builder, first, second, &upper);
            value          = add_word(builder, value, third, &upper);
            if (instruction->carry_in)
            {
                const uint32_t *input  = instruction->predicate_sources;
                const uint32_t *invert = instruction->predicate_source_inverts;
                for (uint32_t index = 0; index < 2U; ++index, ++input, ++invert)
                {
                    value = add_word(builder,
                                     value,
                                     predicate_word(builder, predicate(builder, *input, *invert)),
                                     &upper);
                }
            }
            else
            {
                const uint32_t             values[3] = { first, second, third };
                const uint32_t            *operand   = values;
                const gpu_shader_source_t *original  = instruction->sources;
                for (uint32_t index = 0; index < 3U; ++index, ++operand, ++original)
                {
                    if (original->negate)
                    {
                        upper = binary(
                            builder,
                            SPIRV_ADD,
                            SPIRV_UINT,
                            upper,
                            predicate_word(
                                builder, binary(builder, SPIRV_EQUAL, SPIRV_BOOL, *operand, zero)));
                    }
                }
            }
            predicate_results[0] = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, upper, zero);
            predicate_results[1] = binary(
                builder, SPIRV_UNSIGNED_GREATER_EQUAL, SPIRV_BOOL, upper, constant(builder, 2));
            break;
        }
        case GPU_SHADER_IMAD:
            if (instruction->multiply_mode == GPU_SHADER_MULTIPLY_LOW)
            {
                value = binary(builder,
                               SPIRV_ADD,
                               SPIRV_UINT,
                               binary(builder, SPIRV_MULTIPLY, SPIRV_UINT, first, second),
                               third);
            }
            else
            {
                const uint32_t product
                    = binary(builder,
                             instruction->compare_signed ? SPIRV_MULTIPLY_SIGNED_WIDE
                                                         : SPIRV_MULTIPLY_UNSIGNED_WIDE,
                             SPIRV_PAIR,
                             first,
                             second);
                const uint32_t accumulator = instruction->sources[2].value;
                const uint32_t high        = accumulator == GPU_SHADER_ZERO_REGISTER
                                                 ? zero
                                                 : builder->registers[accumulator + 1U];
                upper_result
                    = binary(builder, SPIRV_ADD, SPIRV_UINT, extract(builder, product, 1), high);
                value = add_word(builder, extract(builder, product, 0), third, &upper_result);
                if (instruction->multiply_mode == GPU_SHADER_MULTIPLY_HIGH)
                {
                    value = upper_result;
                }
            }
            break;
        case GPU_SHADER_IDP4:
        case GPU_SHADER_IDP2:
            value = dot_word(builder, first, second, third, instruction);
            break;
        case GPU_SHADER_SHF: {
            const uint32_t wide = instruction->shift_type == GPU_SHADER_SHIFT_S64
                                  || instruction->shift_type == GPU_SHADER_SHIFT_U64;
            const uint32_t signed_value = instruction->shift_type == GPU_SHADER_SHIFT_S64
                                          || instruction->shift_type == GPU_SHADER_SHIFT_S32;
            value = funnel_word(
                builder,
                first,
                third,
                shift_distance(builder, second, wide ? 64U : 32U, instruction->shift_wrap),
                instruction->shift_right,
                signed_value,
                instruction->shift_high);
            break;
        }
        case GPU_SHADER_FLO:
            value = leading_bit(builder, first, instruction);
            break;
        case GPU_SHADER_PRMT:
            value = permute_word(builder, first, second, third, instruction->permute_mode);
            break;
        case GPU_SHADER_SGXT:
            value = bit_field(builder,
                              first,
                              zero,
                              shift_distance(builder, second, 32U, instruction->shift_wrap),
                              instruction->compare_signed);
            break;
        case GPU_SHADER_BMSK: {
            const uint32_t position = shift_distance(builder, first, 32U, instruction->shift_wrap);
            const uint32_t width    = shift_distance(builder, second, 32U, instruction->shift_wrap);
            const uint32_t full     = constant(builder, 32);
            const uint32_t mask     = select_value(
                builder,
                SPIRV_UINT,
                binary(builder, SPIRV_EQUAL, SPIRV_BOOL, width, full),
                constant(builder, UINT32_MAX),
                binary(builder,
                       SPIRV_SUBTRACT,
                       SPIRV_UINT,
                       binary(builder,
                              SPIRV_SHIFT_LEFT,
                              SPIRV_UINT,
                              constant(builder, 1),
                              binary(builder, SPIRV_AND, SPIRV_UINT, width, constant(builder, 31))),
                       constant(builder, 1)));
            value = select_value(
                builder,
                SPIRV_UINT,
                binary(builder, SPIRV_EQUAL, SPIRV_BOOL, position, full),
                zero,
                binary(builder,
                       SPIRV_SHIFT_LEFT,
                       SPIRV_UINT,
                       mask,
                       binary(builder, SPIRV_AND, SPIRV_UINT, position, constant(builder, 31))));
            break;
        }
        case GPU_SHADER_LEA: {
            const uint32_t shifted
                = instruction->shift_high
                      ? funnel_word(builder,
                                    first,
                                    third,
                                    constant(builder, 32U - instruction->address_shift),
                                    1,
                                    0,
                                    0)
                      : binary(builder,
                               SPIRV_SHIFT_LEFT,
                               SPIRV_UINT,
                               first,
                               constant(builder, instruction->address_shift));
            uint32_t upper = zero;
            value          = instruction->intermediate_negate
                                 ? unary(builder, SPIRV_NOT, SPIRV_UINT, shifted)
                                 : shifted;
            if (instruction->intermediate_negate && !instruction->carry_in)
            {
                value = add_word(builder, value, constant(builder, 1), &upper);
            }
            value = add_word(builder, value, second, &upper);
            if (instruction->carry_in)
            {
                value = add_word(
                    builder,
                    value,
                    predicate_word(
                        builder,
                        predicate(builder, instruction->condition, instruction->condition_invert)),
                    &upper);
            }
            else if (instruction->sources[1].negate)
            {
                upper
                    = binary(builder,
                             SPIRV_ADD,
                             SPIRV_UINT,
                             upper,
                             predicate_word(
                                 builder, binary(builder, SPIRV_EQUAL, SPIRV_BOOL, second, zero)));
            }
            predicate_results[0] = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, upper, zero);
            break;
        }
        case GPU_SHADER_LOP3:
            value = logic_lookup(builder, first, second, third, instruction->lookup);
            predicate_results[0]
                = combine(builder,
                          binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, value, zero),
                          predicate(builder, instruction->condition, instruction->condition_invert),
                          instruction->boolean_operation);
            break;
        case GPU_SHADER_POPC:
            value = unary(builder, SPIRV_COUNT, SPIRV_UINT, first);
            break;
        case GPU_SHADER_SEL:
            value = select_value(
                builder,
                SPIRV_UINT,
                predicate(builder, instruction->condition, instruction->condition_invert),
                first,
                second);
            break;
        case GPU_SHADER_ISETP: {
            uint32_t result = compare(
                builder, first, second, instruction->comparison, instruction->compare_signed);
            const uint32_t condition
                = predicate(builder, instruction->condition, instruction->condition_invert);
            if (instruction->compare_extended && instruction->comparison != GPU_SHADER_COMPARE_FALSE
                && instruction->comparison != GPU_SHADER_COMPARE_TRUE)
            {
                result = select_value(builder,
                                      SPIRV_BOOL,
                                      binary(builder, SPIRV_EQUAL, SPIRV_BOOL, first, second),
                                      predicate(builder,
                                                instruction->predicate_sources[1],
                                                instruction->predicate_source_inverts[1]),
                                      result);
            }
            predicate_results[0]
                = combine(builder, result, condition, instruction->boolean_operation);
            predicate_results[1] = combine(builder,
                                           unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, result),
                                           condition,
                                           instruction->boolean_operation);
            break;
        }
        case GPU_SHADER_FSETP: {
            const uint32_t result
                = compare_float(builder, first, second, instruction->float_comparison);
            const uint32_t condition
                = predicate(builder, instruction->condition, instruction->condition_invert);
            predicate_results[0]
                = combine(builder, result, condition, instruction->boolean_operation);
            predicate_results[1] = combine(builder,
                                           unary(builder, SPIRV_BOOL_NOT, SPIRV_BOOL, result),
                                           condition,
                                           instruction->boolean_operation);
            break;
        }
        case GPU_SHADER_FRND:
            value = float_round_integer(builder, first, instruction);
            break;
        case GPU_SHADER_F2F:
            value = instruction->float_widen
                        ? half_to_float(
                              builder, source(builder, &instruction->sources[0]), instruction)
                        : float_to_half(builder, first, instruction);
            break;
        case GPU_SHADER_F2I:
            value = float_to_integer(builder, first, instruction);
            break;
        case GPU_SHADER_I2F:
        case GPU_SHADER_I2FP:
            value = integer_to_float(builder, first, instruction);
            break;
        case GPU_SHADER_FMNMX:
            value = float_minmax(
                builder,
                first,
                second,
                predicate(builder, instruction->condition, instruction->condition_invert));
            break;
        case GPU_SHADER_FADD:
        case GPU_SHADER_FMUL:
        case GPU_SHADER_FFMA:
            value = float_arithmetic(builder, first, second, third, instruction);
            break;
        case GPU_SHADER_PLOP3: {
            uint32_t        lookup_index = zero;
            const uint32_t *operand      = instruction->predicate_sources;
            const uint32_t *invert       = instruction->predicate_source_inverts;
            for (uint32_t index = 0; index < 3U; ++index, ++operand, ++invert)
            {
                const uint32_t bit = select_value(builder,
                                                  SPIRV_UINT,
                                                  predicate(builder, *operand, *invert),
                                                  constant(builder, 1),
                                                  zero);
                lookup_index       = binary(
                    builder,
                    SPIRV_ADD,
                    SPIRV_UINT,
                    binary(builder, SPIRV_MULTIPLY, SPIRV_UINT, lookup_index, constant(builder, 2)),
                    bit);
            }
            for (uint32_t index = 0; index < 2U; ++index)
            {
                const uint32_t bit
                    = binary(builder,
                             SPIRV_AND,
                             SPIRV_UINT,
                             binary(builder,
                                    SPIRV_SHIFT_RIGHT,
                                    SPIRV_UINT,
                                    constant(builder, instruction->predicate_lookups[index]),
                                    lookup_index),
                             constant(builder, 1));
                predicate_results[index] = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, bit, zero);
            }
            break;
        }
        case GPU_SHADER_IMNMX: {
            const uint32_t less = compare(
                builder, first, second, GPU_SHADER_COMPARE_LT, instruction->compare_signed);
            const uint32_t minimum
                = predicate(builder, instruction->condition, instruction->condition_invert);
            value = select_value(builder,
                                 SPIRV_UINT,
                                 binary(builder, SPIRV_BOOL_EQUAL, SPIRV_BOOL, less, minimum),
                                 first,
                                 second);
            break;
        }
        case GPU_SHADER_BREV:
            value = unary(builder, SPIRV_REVERSE, SPIRV_UINT, first);
            break;
        case GPU_SHADER_IABS:
            value = select_value(builder,
                                 SPIRV_UINT,
                                 compare(builder, first, zero, GPU_SHADER_COMPARE_LT, 1U),
                                 binary(builder, SPIRV_SUBTRACT, SPIRV_UINT, zero, first),
                                 first);
            break;
        default:
            break;
    }
    const uint32_t *destination = instruction->predicate_destinations;
    const uint32_t *result      = predicate_results;
    for (uint32_t index = 0; index < instruction->predicate_destination_count;
         ++index, ++destination, ++result)
    {
        if (*destination != GPU_SHADER_TRUE_PREDICATE)
        {
            builder->predicates[*destination] = select_value(
                builder, SPIRV_BOOL, active, *result, builder->predicates[*destination]);
        }
    }
    if (instruction->source_count && instruction->destination != GPU_SHADER_ZERO_REGISTER)
    {
        builder->registers[instruction->destination]
            = select_word(builder, active, value, builder->registers[instruction->destination]);
        if (instruction->operation == GPU_SHADER_IMAD
            && instruction->multiply_mode == GPU_SHADER_MULTIPLY_WIDE)
        {
            builder->registers[instruction->destination + 1U] = select_word(
                builder, active, upper_result, builder->registers[instruction->destination + 1U]);
        }
    }
}

static void
build_prefix(spirv_builder_t *builder, const gpu_shader_program_t *program)
{
    build_constants(builder, program);
    builder->count = 5U;
    EMIT(builder, SPIRV_CAPABILITY, 1U);
    EMIT(builder, SPIRV_MEMORY_MODEL, 0U, 1U);
    EMIT(builder, SPIRV_ENTRY_POINT, 5U, SPIRV_MAIN, 0x6E69616DU, 0U, SPIRV_GLOBAL_ID);
    EMIT(builder, SPIRV_EXECUTION_MODE, SPIRV_MAIN, 17U, 1U, 1U, 1U);
    EMIT(builder, SPIRV_DECORATE, SPIRV_ARRAY, 6U, 4U);
    EMIT(builder, SPIRV_DECORATE, SPIRV_BUFFER_TYPE, 2U);
    EMIT(builder, SPIRV_MEMBER_DECORATE, SPIRV_BUFFER_TYPE, 0U, 35U, 0U);
    EMIT(builder, SPIRV_DECORATE, SPIRV_BUFFER, 34U, 0U);
    EMIT(builder, SPIRV_DECORATE, SPIRV_BUFFER, 33U, 0U);
    EMIT(builder, SPIRV_DECORATE, SPIRV_GLOBAL_ID, 11U, 28U);
    EMIT(builder, SPIRV_TYPE_VOID, SPIRV_VOID);
    EMIT(builder, SPIRV_TYPE_BOOL, SPIRV_BOOL);
    EMIT(builder, SPIRV_TYPE_INT, SPIRV_UINT, 32U, 0U);
    EMIT(builder, SPIRV_TYPE_INT, SPIRV_INT, 32U, 1U);
    EMIT(builder, SPIRV_TYPE_VECTOR, SPIRV_VECTOR, SPIRV_UINT, 3U);
    EMIT(builder, SPIRV_TYPE_RUNTIME_ARRAY, SPIRV_ARRAY, SPIRV_UINT);
    EMIT(builder, SPIRV_TYPE_STRUCT, SPIRV_BUFFER_TYPE, SPIRV_ARRAY);
    EMIT(builder, SPIRV_TYPE_STRUCT, SPIRV_PAIR, SPIRV_UINT, SPIRV_UINT);
    EMIT(builder, SPIRV_TYPE_POINTER, SPIRV_BUFFER_POINTER, 12U, SPIRV_BUFFER_TYPE);
    EMIT(builder, SPIRV_TYPE_POINTER, SPIRV_WORD_POINTER, 12U, SPIRV_UINT);
    EMIT(builder, SPIRV_TYPE_POINTER, SPIRV_INPUT_POINTER, 1U, SPIRV_VECTOR);
    EMIT(builder, SPIRV_TYPE_POINTER, SPIRV_LOCAL_WORD_POINTER, 7U, SPIRV_UINT);
    EMIT(builder, SPIRV_TYPE_POINTER, SPIRV_LOCAL_BOOL_POINTER, 7U, SPIRV_BOOL);
    EMIT(builder, SPIRV_TYPE_FUNCTION, SPIRV_FUNCTION_TYPE, SPIRV_VOID);
    EMIT(builder, SPIRV_CONSTANT_TRUE, SPIRV_BOOL, SPIRV_TRUE);
    EMIT(builder, SPIRV_CONSTANT_FALSE, SPIRV_BOOL, SPIRV_FALSE);
    const uint32_t *value = builder->constants;
    for (uint32_t index = 0; index < builder->constant_count; ++index, ++value)
    {
        EMIT(builder, SPIRV_CONSTANT, SPIRV_UINT, SPIRV_LOCAL_BOOL_POINTER + 1U + index, *value);
    }
    EMIT(builder, SPIRV_VARIABLE, SPIRV_BUFFER_POINTER, SPIRV_BUFFER, 12U);
    EMIT(builder, SPIRV_VARIABLE, SPIRV_INPUT_POINTER, SPIRV_GLOBAL_ID, 1U);
    EMIT(builder, SPIRV_FUNCTION, SPIRV_VOID, SPIRV_MAIN, 0U, SPIRV_FUNCTION_TYPE);
    EMIT(builder, SPIRV_LABEL, SPIRV_ENTRY);
}

static uint32_t
buffer_pointer(spirv_builder_t *builder, uint32_t offset)
{
    const uint32_t result = builder->next_id++;
    EMIT(builder,
         SPIRV_ACCESS_CHAIN,
         SPIRV_WORD_POINTER,
         result,
         SPIRV_BUFFER,
         constant(builder, 0),
         offset);
    return result;
}

static void
build_end(spirv_builder_t *builder)
{
    emit(builder, SPIRV_RETURN, NULL, 0);
    emit(builder, SPIRV_FUNCTION_END, NULL, 0);
    if (NULL != builder->words)
    {
        const uint32_t  header[5]   = { 0x07230203U, 0x00010300U, 0, builder->next_id, 0 };
        uint32_t       *cursor      = builder->words;
        const uint32_t *header_word = header;
        for (uint32_t index = 0; index < 5U; ++index, ++cursor, ++header_word)
        {
            *cursor = *header_word;
        }
    }
}

static void
build(spirv_builder_t *builder, const gpu_shader_program_t *program)
{
    build_prefix(builder, program);
    const uint32_t invocation = unary(builder, SPIRV_LOAD, SPIRV_VECTOR, SPIRV_GLOBAL_ID);
    const uint32_t base       = binary(builder,
                                 SPIRV_MULTIPLY,
                                 SPIRV_UINT,
                                 extract(builder, invocation, 0),
                                 constant(builder, 256U));
    uint32_t      *pointer    = builder->pointers;
    uint32_t      *operand    = builder->registers;
    uint32_t       mask       = 0;
    for (uint32_t index = 0; index < GPU_SHADER_SPIRV_LANE_WORDS; ++index, ++pointer)
    {
        const uint32_t offset
            = binary(builder, SPIRV_ADD, SPIRV_UINT, base, constant(builder, index));
        *pointer              = buffer_pointer(builder, offset);
        const uint32_t loaded = unary(builder, SPIRV_LOAD, SPIRV_UINT, *pointer);
        if (index == GPU_SHADER_ZERO_REGISTER)
        {
            mask = loaded;
        }
        else
        {
            *operand++ = loaded;
        }
    }
    uint32_t *predicate_value = builder->predicates;
    for (uint32_t index = 0; index < GPU_SHADER_TRUE_PREDICATE; ++index, ++predicate_value)
    {
        const uint32_t bit
            = binary(builder, SPIRV_AND, SPIRV_UINT, mask, constant(builder, 1U << index));
        *predicate_value = binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, bit, constant(builder, 0));
    }
    const gpu_shader_node_t *node = program->nodes;
    for (size_t index = 0; index < program->instruction_count; ++index, ++node)
    {
        emit_instruction(builder, &node->instruction);
    }
    mask            = constant(builder, 0);
    predicate_value = builder->predicates;
    for (uint32_t index = 0; index < GPU_SHADER_TRUE_PREDICATE; ++index, ++predicate_value)
    {
        const uint32_t bit = select_value(builder,
                                          SPIRV_UINT,
                                          *predicate_value,
                                          constant(builder, 1U << index),
                                          constant(builder, 0));
        mask               = binary(builder, SPIRV_OR, SPIRV_UINT, mask, bit);
    }
    pointer = builder->pointers;
    operand = builder->registers;
    for (uint32_t index = 0; index < GPU_SHADER_ZERO_REGISTER; ++index, ++pointer, ++operand)
    {
        EMIT(builder, SPIRV_STORE, *pointer, *operand);
    }
    EMIT(builder, SPIRV_STORE, *pointer, mask);
    build_end(builder);
}

static uint32_t
local_variable(spirv_builder_t *builder, uint32_t type, uint32_t initializer)
{
    const uint32_t result = builder->next_id++;
    if (initializer)
    {
        EMIT(builder, SPIRV_VARIABLE, type, result, 7U, initializer);
    }
    else
    {
        EMIT(builder, SPIRV_VARIABLE, type, result, 7U);
    }
    return result;
}

static uint32_t
load_state(spirv_builder_t *builder, uint32_t word)
{
    return unary(
        builder, SPIRV_LOAD, SPIRV_UINT, builder->state_variables[word - GPU_SHADER_ZERO_REGISTER]);
}

static void
store_state(spirv_builder_t *builder, uint32_t word, uint32_t value)
{
    EMIT(builder, SPIRV_STORE, builder->state_variables[word - GPU_SHADER_ZERO_REGISTER], value);
}

static void
load_register(spirv_builder_t *builder, uint32_t index)
{
    if (index != GPU_SHADER_ZERO_REGISTER)
    {
        builder->registers[index]
            = unary(builder, SPIRV_LOAD, SPIRV_UINT, builder->register_variables[index]);
    }
}

static void
load_node(spirv_builder_t *builder, const gpu_shader_instruction_t *instruction)
{
    const gpu_shader_source_t *operand = instruction->sources;
    for (uint32_t index = 0; index < 3U; ++index, ++operand)
    {
        if (operand->kind == GPU_SHADER_SOURCE_REGISTER)
        {
            load_register(builder, operand->value);
        }
    }
    if (instruction->source_count)
    {
        load_register(builder, instruction->destination);
    }
    if (instruction->operation == GPU_SHADER_IMAD
        && instruction->multiply_mode != GPU_SHADER_MULTIPLY_LOW)
    {
        if (instruction->sources[2].value != GPU_SHADER_ZERO_REGISTER)
        {
            load_register(builder, instruction->sources[2].value + 1U);
        }
        if (instruction->multiply_mode == GPU_SHADER_MULTIPLY_WIDE
            && instruction->destination != GPU_SHADER_ZERO_REGISTER)
        {
            load_register(builder, instruction->destination + 1U);
        }
    }
    uint32_t       *value    = builder->predicates;
    const uint32_t *variable = builder->predicate_variables;
    for (uint32_t index = 0; index < GPU_SHADER_TRUE_PREDICATE; ++index, ++value, ++variable)
    {
        *value = unary(builder, SPIRV_LOAD, SPIRV_BOOL, *variable);
    }
}

static void
store_node(spirv_builder_t *builder, const gpu_shader_instruction_t *instruction)
{
    const uint32_t *destination = instruction->predicate_destinations;
    for (uint32_t index = 0; index < instruction->predicate_destination_count;
         ++index, ++destination)
    {
        if (*destination != GPU_SHADER_TRUE_PREDICATE)
        {
            EMIT(builder,
                 SPIRV_STORE,
                 builder->predicate_variables[*destination],
                 builder->predicates[*destination]);
        }
    }
    if (instruction->source_count && instruction->destination != GPU_SHADER_ZERO_REGISTER)
    {
        const uint32_t target = instruction->destination;
        EMIT(builder, SPIRV_STORE, builder->register_variables[target], builder->registers[target]);
        if (instruction->operation == GPU_SHADER_IMAD
            && instruction->multiply_mode == GPU_SHADER_MULTIPLY_WIDE)
        {
            EMIT(builder,
                 SPIRV_STORE,
                 builder->register_variables[target + 1U],
                 builder->registers[target + 1U]);
        }
    }
}

static void
build_bounded(spirv_builder_t *builder, const gpu_shader_program_t *program)
{
    build_prefix(builder, program);
    const uint32_t zero     = constant(builder, 0);
    uint32_t      *variable = builder->register_variables;
    for (uint32_t index = 0; index < GPU_SHADER_ZERO_REGISTER; ++index, ++variable)
    {
        *variable = local_variable(builder, SPIRV_LOCAL_WORD_POINTER, 0);
    }
    variable = builder->predicate_variables;
    for (uint32_t index = 0; index < GPU_SHADER_TRUE_PREDICATE; ++index, ++variable)
    {
        *variable = local_variable(builder, SPIRV_LOCAL_BOOL_POINTER, 0);
    }
    variable = builder->state_variables;
    for (uint32_t index = 0; index < 6U; ++index, ++variable)
    {
        *variable = local_variable(builder, SPIRV_LOCAL_WORD_POINTER, zero);
    }
    builder->counter_variable = local_variable(builder, SPIRV_LOCAL_WORD_POINTER, zero);
    const uint32_t invocation = unary(builder, SPIRV_LOAD, SPIRV_VECTOR, SPIRV_GLOBAL_ID);
    const uint32_t base       = binary(builder,
                                 SPIRV_MULTIPLY,
                                 SPIRV_UINT,
                                 extract(builder, invocation, 0),
                                 constant(builder, GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS));
    uint32_t      *pointer    = builder->pointers;
    variable                  = builder->register_variables;
    for (uint32_t index = 0; index < GPU_SHADER_SPIRV_BOUNDED_LANE_WORDS; ++index, ++pointer)
    {
        *pointer = buffer_pointer(
            builder, binary(builder, SPIRV_ADD, SPIRV_UINT, base, constant(builder, index)));
        if (index < GPU_SHADER_ZERO_REGISTER)
        {
            EMIT(builder,
                 SPIRV_STORE,
                 *variable++,
                 unary(builder, SPIRV_LOAD, SPIRV_UINT, *pointer));
        }
        else if (index != GPU_SHADER_SPIRV_STATUS)
        {
            store_state(builder, index, unary(builder, SPIRV_LOAD, SPIRV_UINT, *pointer));
        }
    }
    const uint32_t raw_mask = load_state(builder, GPU_SHADER_ZERO_REGISTER);
    variable                = builder->predicate_variables;
    for (uint32_t index = 0; index < GPU_SHADER_TRUE_PREDICATE; ++index, ++variable)
    {
        const uint32_t bit
            = binary(builder, SPIRV_AND, SPIRV_UINT, raw_mask, constant(builder, 1U << index));
        EMIT(builder,
             SPIRV_STORE,
             *variable,
             binary(builder, SPIRV_NOT_EQUAL, SPIRV_BOOL, bit, zero));
    }
    const uint32_t initial_pc = load_state(builder, GPU_SHADER_SPIRV_PC);
    const uint32_t invalid_pc
        = binary(builder,
                 SPIRV_BOOL_OR,
                 SPIRV_BOOL,
                 binary(builder,
                        SPIRV_UNSIGNED_GREATER,
                        SPIRV_BOOL,
                        initial_pc,
                        constant(builder, (uint32_t)program->byte_size)),
                 binary(builder,
                        SPIRV_NOT_EQUAL,
                        SPIRV_BOOL,
                        binary(builder, SPIRV_AND, SPIRV_UINT, initial_pc, constant(builder, 15)),
                        zero));
    const uint32_t invalid_mask
        = binary(builder,
                 SPIRV_NOT_EQUAL,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, raw_mask, constant(builder, 7)),
                 zero);
    const uint32_t invalid = binary(builder,
                                    SPIRV_BOOL_OR,
                                    SPIRV_BOOL,
                                    invalid_pc,
                                    binary(builder,
                                           SPIRV_BOOL_OR,
                                           SPIRV_BOOL,
                                           invalid_mask,
                                           binary(builder,
                                                  SPIRV_UNSIGNED_GREATER,
                                                  SPIRV_BOOL,
                                                  load_state(builder, GPU_SHADER_SPIRV_HALTED),
                                                  constant(builder, 1))));
    store_state(
        builder,
        GPU_SHADER_SPIRV_STATUS,
        select_value(
            builder, SPIRV_UINT, invalid, constant(builder, POUND_ERROR_INVALID_ARGUMENT), zero));
    uint32_t       next_id      = builder->next_id;
    const uint32_t header       = next_id++;
    const uint32_t check        = next_id++;
    const uint32_t failed       = next_id++;
    const uint32_t dispatch     = next_id++;
    const uint32_t default_case = next_id++;
    const uint32_t switch_merge = next_id++;
    const uint32_t check_merge  = next_id++;
    const uint32_t continuing   = next_id++;
    const uint32_t done         = next_id++;
    uint32_t       labels[GPU_SHADER_SPIRV_INSTRUCTION_LIMIT];
    uint32_t      *label = labels;
    for (size_t index = 0; index < program->instruction_count; ++index, ++label)
    {
        *label = next_id++;
    }
    builder->next_id = next_id;
    EMIT(builder, SPIRV_BRANCH, header);
    EMIT(builder, SPIRV_LABEL, header);
    const uint32_t counter = unary(builder, SPIRV_LOAD, SPIRV_UINT, builder->counter_variable);
    const uint32_t running = binary(
        builder,
        SPIRV_BOOL_AND,
        SPIRV_BOOL,
        binary(
            builder, SPIRV_UNSIGNED_LESS, SPIRV_BOOL, counter, constant(builder, builder->budget)),
        binary(builder,
               SPIRV_BOOL_AND,
               SPIRV_BOOL,
               binary(builder,
                      SPIRV_EQUAL,
                      SPIRV_BOOL,
                      load_state(builder, GPU_SHADER_SPIRV_HALTED),
                      zero),
               binary(builder,
                      SPIRV_EQUAL,
                      SPIRV_BOOL,
                      load_state(builder, GPU_SHADER_SPIRV_STATUS),
                      zero)));
    EMIT(builder, SPIRV_LOOP_MERGE, done, continuing, 0U);
    EMIT(builder, SPIRV_BRANCH_CONDITIONAL, running, check, done);
    EMIT(builder, SPIRV_LABEL, check);
    const uint32_t pc   = load_state(builder, GPU_SHADER_SPIRV_PC);
    const uint32_t low  = load_state(builder, GPU_SHADER_SPIRV_STEPS_LOW);
    const uint32_t high = load_state(builder, GPU_SHADER_SPIRV_STEPS_HIGH);
    const uint32_t overflow
        = binary(builder,
                 SPIRV_BOOL_AND,
                 SPIRV_BOOL,
                 binary(builder, SPIRV_EQUAL, SPIRV_BOOL, low, constant(builder, UINT32_MAX)),
                 binary(builder, SPIRV_EQUAL, SPIRV_BOOL, high, constant(builder, UINT32_MAX)));
    const uint32_t failure
        = select_value(builder,
                       SPIRV_UINT,
                       binary(builder,
                              SPIRV_UNSIGNED_GREATER_EQUAL,
                              SPIRV_BOOL,
                              pc,
                              constant(builder, (uint32_t)program->byte_size)),
                       constant(builder, POUND_ERROR_GPU_TRUNCATED_COMMAND),
                       select_value(builder,
                                    SPIRV_UINT,
                                    overflow,
                                    constant(builder, POUND_ERROR_GPU_RESOURCE_LIMIT),
                                    zero));
    store_state(builder, GPU_SHADER_SPIRV_STATUS, failure);
    const uint32_t ready = binary(builder, SPIRV_EQUAL, SPIRV_BOOL, failure, zero);
    EMIT(builder, SPIRV_SELECTION_MERGE, check_merge, 0U);
    EMIT(builder, SPIRV_BRANCH_CONDITIONAL, ready, dispatch, failed);
    EMIT(builder, SPIRV_LABEL, failed);
    EMIT(builder, SPIRV_BRANCH, check_merge);
    EMIT(builder, SPIRV_LABEL, dispatch);
    const uint32_t instruction_index
        = binary(builder, SPIRV_SHIFT_RIGHT, SPIRV_UINT, pc, constant(builder, 4));
    uint32_t  operands[2U + GPU_SHADER_SPIRV_INSTRUCTION_LIMIT * 2U];
    uint32_t *operand = operands;
    *operand++        = instruction_index;
    *operand++        = default_case;
    label             = labels;
    for (size_t index = 0; index < program->instruction_count; ++index, ++label)
    {
        *operand++ = (uint32_t)index;
        *operand++ = *label;
    }
    EMIT(builder, SPIRV_SELECTION_MERGE, switch_merge, 0U);
    emit(builder, SPIRV_SWITCH, operands, 2U + program->instruction_count * 2U);
    const gpu_shader_node_t *node = program->nodes;
    label                         = labels;
    for (size_t index = 0; index < program->instruction_count; ++index, ++node, ++label)
    {
        EMIT(builder, SPIRV_LABEL, *label);
        const gpu_shader_instruction_t *instruction = &node->instruction;
        load_node(builder, instruction);
        uint32_t following
            = constant(builder, (uint32_t)((index + 1U) * GPU_SHADER_INSTRUCTION_SIZE));
        const uint32_t active
            = predicate(builder, instruction->predicate, instruction->predicate_invert);
        if (instruction->operation == GPU_SHADER_BRA)
        {
            const uint32_t taken
                = binary(builder,
                         SPIRV_BOOL_AND,
                         SPIRV_BOOL,
                         active,
                         predicate(builder, instruction->condition, instruction->condition_invert));
            following = select_value(
                builder,
                SPIRV_UINT,
                taken,
                constant(builder,
                         (uint32_t)(node->branch_instruction * GPU_SHADER_INSTRUCTION_SIZE)),
                following);
        }
        else if (instruction->operation == GPU_SHADER_EXIT)
        {
            store_state(builder,
                        GPU_SHADER_SPIRV_HALTED,
                        select_value(builder,
                                     SPIRV_UINT,
                                     active,
                                     constant(builder, 1),
                                     load_state(builder, GPU_SHADER_SPIRV_HALTED)));
        }
        else
        {
            emit_instruction(builder, instruction);
            store_node(builder, instruction);
        }
        store_state(builder, GPU_SHADER_SPIRV_PC, following);
        EMIT(builder, SPIRV_BRANCH, switch_merge);
    }
    EMIT(builder, SPIRV_LABEL, default_case);
    store_state(
        builder, GPU_SHADER_SPIRV_STATUS, constant(builder, POUND_ERROR_GPU_TRUNCATED_COMMAND));
    EMIT(builder, SPIRV_BRANCH, switch_merge);
    EMIT(builder, SPIRV_LABEL, switch_merge);
    const uint32_t executed = binary(
        builder, SPIRV_EQUAL, SPIRV_BOOL, load_state(builder, GPU_SHADER_SPIRV_STATUS), zero);
    uint32_t       upper = high;
    const uint32_t step  = add_word(builder, low, predicate_word(builder, executed), &upper);
    store_state(builder, GPU_SHADER_SPIRV_STEPS_LOW, step);
    store_state(builder, GPU_SHADER_SPIRV_STEPS_HIGH, upper);
    EMIT(builder, SPIRV_BRANCH, check_merge);
    EMIT(builder, SPIRV_LABEL, check_merge);
    EMIT(builder, SPIRV_BRANCH, continuing);
    EMIT(builder, SPIRV_LABEL, continuing);
    EMIT(builder,
         SPIRV_STORE,
         builder->counter_variable,
         binary(builder, SPIRV_ADD, SPIRV_UINT, counter, constant(builder, 1)));
    EMIT(builder, SPIRV_BRANCH, header);
    EMIT(builder, SPIRV_LABEL, done);
    uint32_t       status    = load_state(builder, GPU_SHADER_SPIRV_STATUS);
    const uint32_t exhausted = binary(
        builder,
        SPIRV_BOOL_AND,
        SPIRV_BOOL,
        binary(builder, SPIRV_EQUAL, SPIRV_BOOL, status, zero),
        binary(
            builder, SPIRV_EQUAL, SPIRV_BOOL, load_state(builder, GPU_SHADER_SPIRV_HALTED), zero));
    status = select_value(
        builder, SPIRV_UINT, exhausted, constant(builder, POUND_ERROR_GPU_RESOURCE_LIMIT), status);
    store_state(builder, GPU_SHADER_SPIRV_STATUS, status);
    pointer  = builder->pointers;
    variable = builder->register_variables;
    for (uint32_t index = 0; index < GPU_SHADER_ZERO_REGISTER; ++index, ++pointer, ++variable)
    {
        EMIT(builder, SPIRV_STORE, *pointer, unary(builder, SPIRV_LOAD, SPIRV_UINT, *variable));
    }
    uint32_t mask = zero;
    variable      = builder->predicate_variables;
    for (uint32_t index = 0; index < GPU_SHADER_TRUE_PREDICATE; ++index, ++variable)
    {
        mask = binary(builder,
                      SPIRV_OR,
                      SPIRV_UINT,
                      mask,
                      select_value(builder,
                                   SPIRV_UINT,
                                   unary(builder, SPIRV_LOAD, SPIRV_BOOL, *variable),
                                   constant(builder, 1U << index),
                                   zero));
    }
    EMIT(builder,
         SPIRV_STORE,
         *pointer++,
         select_value(builder, SPIRV_UINT, invalid, raw_mask, mask));
    for (uint32_t word = GPU_SHADER_SPIRV_PC; word <= GPU_SHADER_SPIRV_STATUS; ++word, ++pointer)
    {
        EMIT(builder, SPIRV_STORE, *pointer, load_state(builder, word));
    }
    build_end(builder);
}

error_t
gpu_shader_spirv_emit(const gpu_shader_program_t *program,
                      uint32_t                   *words,
                      size_t                      capacity,
                      size_t                     *word_count)
{
    if (NULL == word_count || (NULL == words && capacity != 0))
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const error_t status = validate(program, 0);
    if (status != POUND_SUCCESS)
    {
        return status;
    }
    spirv_builder_t builder = { 0 };
    build(&builder, program);
    const size_t count = builder.count;
    if (NULL != words)
    {
        if (capacity < count)
        {
            return POUND_ERROR_GPU_RESOURCE_LIMIT;
        }
        builder = (spirv_builder_t) { .words = words };
        build(&builder, program);
    }
    *word_count = count;
    return POUND_SUCCESS;
}

error_t
gpu_shader_spirv_emit_bounded(const gpu_shader_program_t *program,
                              uint32_t                    budget,
                              uint32_t                   *words,
                              size_t                      capacity,
                              size_t                     *word_count)
{
    if (NULL == word_count || (NULL == words && capacity != 0) || 0 == budget
        || budget > GPU_SHADER_SPIRV_BUDGET_LIMIT)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const error_t status = validate(program, 1);
    if (status != POUND_SUCCESS)
    {
        return status;
    }
    spirv_builder_t builder = { .budget = budget };
    build_bounded(&builder, program);
    const size_t count = builder.count;
    if (NULL != words)
    {
        if (capacity < count)
        {
            return POUND_ERROR_GPU_RESOURCE_LIMIT;
        }
        builder = (spirv_builder_t) { .words = words, .budget = budget };
        build_bounded(&builder, program);
    }
    *word_count = count;
    return POUND_SUCCESS;
}
