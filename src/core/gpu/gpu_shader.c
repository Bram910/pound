#include "gpu_shader.h"
#include <string.h>

static gpu_shader_source_t
register_source(uint32_t value, uint32_t negate)
{
    return (gpu_shader_source_t) { .kind   = GPU_SHADER_SOURCE_REGISTER,
                                   .value  = value,
                                   .negate = negate };
}

static gpu_shader_source_t
immediate_source(uint32_t value)
{
    return (gpu_shader_source_t) { .kind = GPU_SHADER_SOURCE_IMMEDIATE, .value = value };
}

error_t
gpu_shader_decode(const void *bytes, size_t size, gpu_shader_instruction_t *instruction)
{
    if (NULL == bytes || NULL == instruction || GPU_SHADER_INSTRUCTION_SIZE != size)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const uint8_t           *cursor  = bytes;
    gpu_shader_instruction_t decoded = { 0 };
    uint32_t                *word    = decoded.words;
    for (uint32_t index = 0; index < 4U; ++index, ++word, cursor += 4U)
    {
        *word = (uint32_t)cursor[0] | ((uint32_t)cursor[1] << 8U) | ((uint32_t)cursor[2] << 16U)
                | ((uint32_t)cursor[3] << 24U);
    }
    const uint32_t first     = decoded.words[0];
    const uint32_t second    = decoded.words[1];
    const uint32_t third     = decoded.words[2];
    const uint32_t control   = decoded.words[3];
    decoded.predicate        = (first >> 12U) & 7U;
    decoded.predicate_invert = (first >> 15U) & 1U;
    decoded.destination      = (first >> 16U) & 255U;
    decoded.condition        = (third >> 23U) & 7U;
    decoded.condition_invert = (third >> 26U) & 1U;
    decoded.stall            = (control >> 9U) & 15U;
    decoded.yield            = (control >> 13U) & 1U;
    decoded.write_barrier    = (control >> 14U) & 7U;
    decoded.read_barrier     = (control >> 17U) & 7U;
    decoded.wait_mask        = (control >> 20U) & 63U;
    decoded.reuse_mask       = (control >> 26U) & 15U;
    if (0 != (control & 0xC00001FFU) || 6U == decoded.write_barrier || 6U == decoded.read_barrier)
    {
        return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
    }
    const uint32_t opcode = first & 0xFFFU;
    if (0x918U == opcode || 0x947U == opcode || 0x94DU == opcode)
    {
        if (0 != (first & 0xFFFF0000U))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        if (0x918U == opcode)
        {
            if (0 != second || 0 != third)
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation = GPU_SHADER_NOP;
        }
        else if (0x94DU == opcode)
        {
            if (0 != second || 0x03800000U != third)
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation = GPU_SHADER_EXIT;
        }
        else
        {
            if (0 != (second & 3U) || 0 != (third & ~0x0783FFFFU))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            const uint64_t relative        = (second >> 2U) | ((uint64_t)(third & 0x3FFFFU) << 30U);
            const int64_t  signed_relative = 0 != (relative & (UINT64_C(1) << 47U))
                                                 ? -(int64_t)((UINT64_C(1) << 48U) - relative)
                                                 : (int64_t)relative;
            if (0 != signed_relative % 4)
            {
                return POUND_ERROR_GPU_INVALID_SHADER;
            }
            decoded.operation    = GPU_SHADER_BRA;
            decoded.branch_bytes = signed_relative * 4;
        }
        *instruction = decoded;
        return POUND_SUCCESS;
    }

    if (0x81CU == opcode)
    {
        if (0 != (first & 0xFF000000U) || 0 != second || 0 != (third & ~0x07FFFFF7U))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        decoded.operation                   = GPU_SHADER_PLOP3;
        decoded.destination                 = GPU_SHADER_ZERO_REGISTER;
        decoded.predicate_lookups[0]        = (third & 7U) | (((third >> 8U) & 31U) << 3U);
        decoded.predicate_lookups[1]        = (first >> 16U) & 255U;
        decoded.predicate_sources[0]        = decoded.condition;
        decoded.predicate_source_inverts[0] = decoded.condition_invert;
        decoded.predicate_sources[1]        = (third >> 13U) & 7U;
        decoded.predicate_source_inverts[1] = (third >> 16U) & 1U;
        decoded.predicate_sources[2]        = (third >> 4U) & 7U;
        decoded.predicate_source_inverts[2] = (third >> 7U) & 1U;
        decoded.predicate_source_count      = 3U;
        decoded.predicate_destination_count = 2U;
        decoded.predicate_destinations[0]   = (third >> 17U) & 7U;
        decoded.predicate_destinations[1]   = (third >> 20U) & 7U;
        if (decoded.predicate_destinations[0] == decoded.predicate_destinations[1]
            && GPU_SHADER_TRUE_PREDICATE != decoded.predicate_destinations[0])
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        *instruction = decoded;
        return POUND_SUCCESS;
    }

    const uint32_t operation = opcode & 0x1FFU;
    const uint32_t form      = opcode >> 9U;
    if (1U != form && 4U != form
        && !(2U == form
             && (0x24U == operation || 0x19U == operation || 0x16U == operation
                 || 0x11U == operation)))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
    }
    const uint32_t source_modifiers
        = 0x10U == operation || 0x109U == operation || 0x100U == operation || 0x11U == operation;
    if (1U == form && 0 != (second & ~(0 != source_modifiers ? 0x800000FFU : 0xFFU)))
    {
        return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
    }
    const gpu_shader_source_t middle
        = 1U == form ? register_source(second & 255U, 0 != source_modifiers ? second >> 31U : 0)
                     : immediate_source(second);
    decoded.sources[0] = register_source(first >> 24U, 0);
    decoded.sources[1] = middle;
    decoded.sources[2] = register_source(third & 255U, 0);
    switch (operation)
    {
        case 0x002U:
        case 0x109U:
        case 0x101U:
        case 0x013U:
            if (0 != (first & 0xFF000000U) || (0x002U == operation ? 0xF00U != third : 0 != third))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation
                = 0x002U == operation
                      ? GPU_SHADER_MOV
                      : (0x109U == operation
                             ? GPU_SHADER_POPC
                             : (0x101U == operation ? GPU_SHADER_BREV : GPU_SHADER_IABS));
            decoded.sources[0]   = middle;
            decoded.source_count = 1U;
            break;
        case 0x010U:
            if (0 != (third & ~0x07FFEDFFU)
                || (0 == (third & 0x400U) && 0x0781E000U != (third & 0x0781E000U)))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation                   = GPU_SHADER_IADD3;
            decoded.sources[0].negate           = (third >> 8U) & 1U;
            decoded.sources[2].negate           = (third >> 11U) & 1U;
            decoded.source_count                = 3U;
            decoded.carry_in                    = (third >> 10U) & 1U;
            decoded.predicate_destination_count = 2U;
            decoded.predicate_destinations[0]   = (third >> 17U) & 7U;
            decoded.predicate_destinations[1]   = (third >> 20U) & 7U;
            if (decoded.predicate_destinations[0] == decoded.predicate_destinations[1]
                && GPU_SHADER_TRUE_PREDICATE != decoded.predicate_destinations[0])
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            if (0 != decoded.carry_in)
            {
                decoded.predicate_sources[0]        = decoded.condition;
                decoded.predicate_source_inverts[0] = decoded.condition_invert;
                decoded.predicate_sources[1]        = (third >> 13U) & 7U;
                decoded.predicate_source_inverts[1] = (third >> 16U) & 1U;
                decoded.predicate_source_count      = 2U;
            }
            break;
        case 0x024U:
            if (0x000E0000U != (third & ~0x2FFU))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_IMAD;
            decoded.source_count = 3U;
            break;
        case 0x011U:
            if (0 != (third & ~0x078FFDFFU) || (0 == (third & 0x400U) && 0 != (third & 0x07800000U))
                || (0 == (third & 0x10000U) && (2U == form || 255U != (third & 255U)))
                || (0 != (third & 0x100U) && 0 != middle.negate))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation                   = GPU_SHADER_LEA;
            decoded.address_shift               = (third >> 11U) & 31U;
            decoded.intermediate_negate         = (third >> 8U) & 1U;
            decoded.carry_in                    = (third >> 10U) & 1U;
            decoded.shift_high                  = (third >> 16U) & 1U;
            decoded.source_count                = 0 != decoded.shift_high ? 3U : 2U;
            decoded.predicate_destinations[0]   = (third >> 17U) & 7U;
            decoded.predicate_destination_count = 1U;
            if (0 != decoded.carry_in)
            {
                decoded.predicate_sources[0]        = decoded.condition;
                decoded.predicate_source_inverts[0] = decoded.condition_invert;
                decoded.predicate_source_count      = 1U;
            }
            break;
        case 0x012U:
            if (0x078E0000U != (third & ~0xFFFFU))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_LOP3;
            decoded.lookup       = (third >> 8U) & 255U;
            decoded.source_count = 3U;
            break;
        case 0x007U:
            if (0 != (third & ~0x07800000U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_SEL;
            decoded.source_count = 2U;
            break;
        case 0x00CU:
            if (0 != (first & 0x00FF0000U) || 0 != (third & ~0x07FE7EF0U)
                || 0x70U != (third & 0x70U) || 3U == ((third >> 10U) & 3U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation                   = GPU_SHADER_ISETP;
            decoded.destination                 = GPU_SHADER_ZERO_REGISTER;
            decoded.comparison                  = (gpu_shader_comparison_t)((third >> 12U) & 7U);
            decoded.boolean_operation           = (gpu_shader_boolean_t)((third >> 10U) & 3U);
            decoded.compare_signed              = (third >> 9U) & 1U;
            decoded.source_count                = 2U;
            decoded.predicate_sources[0]        = decoded.condition;
            decoded.predicate_source_inverts[0] = decoded.condition_invert;
            decoded.predicate_source_count      = 1U;
            decoded.predicate_destination_count = 2U;
            decoded.predicate_destinations[0]   = (third >> 17U) & 7U;
            decoded.predicate_destinations[1]   = (third >> 20U) & 7U;
            if (decoded.predicate_destinations[0] == decoded.predicate_destinations[1]
                && GPU_SHADER_TRUE_PREDICATE != decoded.predicate_destinations[0])
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            break;
        case 0x017U:
            if (0 != (third & ~0x07800200U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation      = GPU_SHADER_IMNMX;
            decoded.compare_signed = (third >> 9U) & 1U;
            decoded.source_count   = 2U;
            break;
        case 0x019U:
            if (0 != (third & ~0x00011EFFU))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_SHF;
            decoded.shift_type   = (gpu_shader_shift_type_t)((third >> 9U) & 3U);
            decoded.shift_wrap   = (third >> 11U) & 1U;
            decoded.shift_right  = (third >> 12U) & 1U;
            decoded.shift_high   = (third >> 16U) & 1U;
            decoded.source_count = 3U;
            break;
        case 0x100U:
            if (0 != (first & 0xFF000000U) || 0x000E0000U != (third & ~0x600U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation      = GPU_SHADER_FLO;
            decoded.sources[0]     = middle;
            decoded.source_count   = 1U;
            decoded.compare_signed = (third >> 9U) & 1U;
            decoded.return_shift   = (third >> 10U) & 1U;
            break;
        case 0x016U:
            if (0 != (third & ~0x7FFU) || 7U == ((third >> 8U) & 7U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_PRMT;
            decoded.source_count = 3U;
            decoded.permute_mode = (gpu_shader_permute_mode_t)((third >> 8U) & 7U);
            break;
        case 0x01AU:
            if (0 != (third & ~0xA00U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation      = GPU_SHADER_SGXT;
            decoded.source_count   = 2U;
            decoded.compare_signed = (third >> 9U) & 1U;
            decoded.shift_wrap     = (third >> 11U) & 1U;
            break;
        case 0x01BU:
            if (0 != (third & ~0x800U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_BMSK;
            decoded.source_count = 2U;
            decoded.shift_wrap   = (third >> 11U) & 1U;
            break;
        default:
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
    }
    if (2U == form)
    {
        decoded.sources[1] = decoded.sources[2];
        decoded.sources[2] = immediate_source(second);
    }
    if (GPU_SHADER_IADD3 == decoded.operation && 0 != decoded.sources[0].negate
        && 0 != decoded.sources[1].negate)
    {
        return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
    }
    *instruction = decoded;
    return POUND_SUCCESS;
}

const char *
gpu_shader_operation_name(gpu_shader_operation_t operation)
{
    switch (operation)
    {
        case GPU_SHADER_MOV:
            return "MOV";
        case GPU_SHADER_IADD3:
            return "IADD3";
        case GPU_SHADER_IMAD:
            return "IMAD";
        case GPU_SHADER_LOP3:
            return "LOP3";
        case GPU_SHADER_POPC:
            return "POPC";
        case GPU_SHADER_SEL:
            return "SEL";
        case GPU_SHADER_NOP:
            return "NOP";
        case GPU_SHADER_BRA:
            return "BRA";
        case GPU_SHADER_EXIT:
            return "EXIT";
        case GPU_SHADER_ISETP:
            return "ISETP";
        case GPU_SHADER_PLOP3:
            return "PLOP3";
        case GPU_SHADER_IMNMX:
            return "IMNMX";
        case GPU_SHADER_BREV:
            return "BREV";
        case GPU_SHADER_IABS:
            return "IABS";
        case GPU_SHADER_SHF:
            return "SHF";
        case GPU_SHADER_FLO:
            return "FLO";
        case GPU_SHADER_PRMT:
            return "PRMT";
        case GPU_SHADER_SGXT:
            return "SGXT";
        case GPU_SHADER_BMSK:
            return "BMSK";
        case GPU_SHADER_LEA:
            return "LEA";
        default:
            return "UNKNOWN";
    }
}

error_t
gpu_shader_lane_init(gpu_shader_lane_t *lane)
{
    if (NULL == lane)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    memset(lane, 0, sizeof(*lane));
    return POUND_SUCCESS;
}

static uint32_t
predicate_value(const gpu_shader_lane_t *lane, uint32_t predicate, uint32_t invert)
{
    return (GPU_SHADER_TRUE_PREDICATE == predicate ? 1U : (lane->predicates >> predicate) & 1U)
           ^ invert;
}

static uint32_t
source_value(const gpu_shader_lane_t *lane, const gpu_shader_source_t *source, uint32_t bitwise)
{
    const uint32_t value
        = GPU_SHADER_SOURCE_IMMEDIATE == source->kind
              ? source->value
              : (GPU_SHADER_ZERO_REGISTER == source->value ? 0 : lane->registers[source->value]);
    return 0 == source->negate ? value : (0 != bitwise ? ~value : 0U - value);
}

static uint32_t
compare_integer(uint32_t first, uint32_t second, const gpu_shader_instruction_t *instruction)
{
    if (0 != instruction->compare_signed)
    {
        first ^= 0x80000000U;
        second ^= 0x80000000U;
    }
    switch (instruction->comparison)
    {
        case GPU_SHADER_COMPARE_FALSE:
            return 0;
        case GPU_SHADER_COMPARE_LT:
            return first < second;
        case GPU_SHADER_COMPARE_EQ:
            return first == second;
        case GPU_SHADER_COMPARE_LE:
            return first <= second;
        case GPU_SHADER_COMPARE_GT:
            return first > second;
        case GPU_SHADER_COMPARE_NE:
            return first != second;
        case GPU_SHADER_COMPARE_GE:
            return first >= second;
        case GPU_SHADER_COMPARE_TRUE:
            return 1U;
        default:
            return 0;
    }
}

static uint32_t
combine_predicates(uint32_t first, uint32_t second, gpu_shader_boolean_t operation)
{
    switch (operation)
    {
        case GPU_SHADER_BOOLEAN_AND:
            return first & second;
        case GPU_SHADER_BOOLEAN_OR:
            return first | second;
        case GPU_SHADER_BOOLEAN_XOR:
            return first ^ second;
        default:
            return 0;
    }
}

static uint32_t
funnel_shift(uint32_t                        low,
             uint32_t                        amount,
             uint32_t                        high,
             const gpu_shader_instruction_t *instruction)
{
    const uint32_t wide = GPU_SHADER_SHIFT_S64 == instruction->shift_type
                          || GPU_SHADER_SHIFT_U64 == instruction->shift_type;
    const uint32_t width = 0 != wide ? 64U : 32U;
    const uint32_t distance
        = 0 != instruction->shift_wrap ? amount & (width - 1U) : (amount < width ? amount : width);
    const uint64_t pair = (uint64_t)low | ((uint64_t)high << 32U);
    uint64_t       shifted;
    if (0 != instruction->shift_right)
    {
        const uint32_t sign = (GPU_SHADER_SHIFT_S64 == instruction->shift_type
                               || GPU_SHADER_SHIFT_S32 == instruction->shift_type)
                              && 0 != (high & 0x80000000U);
        shifted = 64U == distance ? 0 : pair >> distance;
        if (0 != sign && 0 != distance)
        {
            shifted |= UINT64_MAX << (64U - distance);
        }
    }
    else
    {
        shifted = 64U == distance ? 0 : pair << distance;
    }
    return (uint32_t)(0 != instruction->shift_high ? shifted >> 32U : shifted);
}

static uint32_t
find_leading(uint32_t value, const gpu_shader_instruction_t *instruction)
{
    if (0 != instruction->compare_signed && 0 != (value & 0x80000000U))
    {
        value = ~value;
    }
    if (0 == value)
    {
        return UINT32_MAX;
    }
    uint32_t leading = 0;
    for (uint32_t mask = 0x80000000U; 0 == (value & mask); mask >>= 1U)
    {
        ++leading;
    }
    return 0 != instruction->return_shift ? leading : 31U - leading;
}

static uint32_t
permute_bytes(uint32_t low, uint32_t selector, uint32_t high, gpu_shader_permute_mode_t mode)
{
    const uint64_t pair  = (uint64_t)low | ((uint64_t)high << 32U);
    uint32_t       value = 0;
    for (uint32_t octet = 0; octet < 4U; ++octet)
    {
        uint32_t selection = selector & 3U;
        switch (mode)
        {
            case GPU_SHADER_PERMUTE_INDEX:
                selection = (selector >> (4U * octet)) & 15U;
                break;
            case GPU_SHADER_PERMUTE_FORWARD4:
                selection += octet;
                break;
            case GPU_SHADER_PERMUTE_BACKWARD4:
                selection = (selection + 8U - octet) & 7U;
                break;
            case GPU_SHADER_PERMUTE_REPLICATE8:
                break;
            case GPU_SHADER_PERMUTE_CLAMP_LEFT:
                selection = octet < selection ? selection : octet;
                break;
            case GPU_SHADER_PERMUTE_CLAMP_RIGHT:
                selection = octet > selection ? selection : octet;
                break;
            case GPU_SHADER_PERMUTE_REPLICATE16:
                selection = 2U * (selection & 1U) + (octet & 1U);
                break;
            default:
                return 0;
        }
        uint32_t selected = (uint32_t)(pair >> (8U * (selection & 7U))) & 255U;
        if (0 != (selection & 8U))
        {
            selected = 0 != (selected & 128U) ? 255U : 0;
        }
        value |= selected << (8U * octet);
    }
    return value;
}

static uint32_t
extract_low_bits(uint32_t value, uint32_t bits, uint32_t signed_value)
{
    if (bits >= 32U)
    {
        return value;
    }
    if (0 == bits)
    {
        return 0;
    }
    const uint32_t mask = (1U << bits) - 1U;
    value &= mask;
    return 0 != signed_value && 0 != (value & (1U << (bits - 1U))) ? value | ~mask : value;
}

error_t
gpu_shader_run(gpu_shader_lane_t *lane, const void *program, size_t size, uint32_t budget)
{
    if (NULL == lane || NULL == program || 0 != (lane->predicates & ~0x7FU)
        || 0 != size % GPU_SHADER_INSTRUCTION_SIZE || 0 != lane->pc % GPU_SHADER_INSTRUCTION_SIZE
        || lane->pc > size || lane->halted > 1U)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    const uint8_t *bytes  = program;
    size_t         pc     = lane->pc;
    uint64_t       steps  = lane->steps;
    uint32_t       halted = lane->halted;
    for (uint32_t count = 0; count < budget && 0 == halted; ++count)
    {
        if (GPU_SHADER_INSTRUCTION_SIZE > size - pc)
        {
            return POUND_ERROR_GPU_TRUNCATED_COMMAND;
        }
        if (UINT64_MAX == steps)
        {
            return POUND_ERROR_GPU_RESOURCE_LIMIT;
        }
        gpu_shader_instruction_t instruction;
        const error_t            status
            = gpu_shader_decode(bytes + pc, GPU_SHADER_INSTRUCTION_SIZE, &instruction);
        if (POUND_SUCCESS != status)
        {
            return status;
        }
        size_t next = pc + GPU_SHADER_INSTRUCTION_SIZE;
        if (0 != predicate_value(lane, instruction.predicate, instruction.predicate_invert))
        {
            uint32_t       value                = 0;
            uint32_t       predicate_results[2] = { 0 };
            const uint32_t complement           = GPU_SHADER_POPC == instruction.operation
                                        || GPU_SHADER_FLO == instruction.operation
                                        || ((GPU_SHADER_IADD3 == instruction.operation
                                             || GPU_SHADER_LEA == instruction.operation)
                                            && 0 != instruction.carry_in);
            const uint32_t first  = source_value(lane, &instruction.sources[0], complement);
            const uint32_t second = source_value(lane, &instruction.sources[1], complement);
            const uint32_t third  = source_value(lane, &instruction.sources[2], complement);
            switch (instruction.operation)
            {
                case GPU_SHADER_MOV:
                    value = first;
                    break;
                case GPU_SHADER_IADD3: {
                    uint64_t sum = (uint64_t)first + second + third;
                    if (0 != instruction.carry_in)
                    {
                        for (uint32_t index = 0; index < 2U; ++index)
                        {
                            sum += predicate_value(lane,
                                                   instruction.predicate_sources[index],
                                                   instruction.predicate_source_inverts[index]);
                        }
                    }
                    else
                    {
                        const uint32_t             values[3] = { first, second, third };
                        const uint32_t            *operand   = values;
                        const gpu_shader_source_t *source    = instruction.sources;
                        for (uint32_t index = 0; index < 3U; ++index, ++operand, ++source)
                        {
                            if (0 != source->negate && 0 == *operand)
                            {
                                sum += UINT64_C(1) << 32U;
                            }
                        }
                    }
                    value                = (uint32_t)sum;
                    predicate_results[0] = sum >= (UINT64_C(1) << 32U);
                    predicate_results[1] = sum >= (UINT64_C(2) << 32U);
                    break;
                }
                case GPU_SHADER_IMAD:
                    value = first * second + third;
                    break;
                case GPU_SHADER_LEA: {
                    const uint32_t shifted
                        = 0 != instruction.shift_high
                              ? (uint32_t)(((uint64_t)first | ((uint64_t)third << 32U))
                                           >> (32U - instruction.address_shift))
                              : first << instruction.address_shift;
                    uint64_t sum = shifted;
                    if (0 != instruction.intermediate_negate)
                    {
                        sum = (uint64_t)~shifted + (0 != instruction.carry_in ? 0U : 1U);
                    }
                    sum += second;
                    if (0 != instruction.carry_in)
                    {
                        sum += predicate_value(
                            lane, instruction.condition, instruction.condition_invert);
                    }
                    else if (0 != instruction.sources[1].negate && 0 == second)
                    {
                        sum += UINT64_C(1) << 32U;
                    }
                    value                = (uint32_t)sum;
                    predicate_results[0] = sum > UINT32_MAX;
                    break;
                }
                case GPU_SHADER_LOP3:
                    for (uint32_t bit = 0; bit < 32U; ++bit)
                    {
                        const uint32_t index = (((first >> bit) & 1U) << 2U)
                                               | (((second >> bit) & 1U) << 1U)
                                               | ((third >> bit) & 1U);
                        value |= ((instruction.lookup >> index) & 1U) << bit;
                    }
                    break;
                case GPU_SHADER_POPC:
                    for (uint32_t remaining = first; 0 != remaining; remaining &= remaining - 1U)
                    {
                        ++value;
                    }
                    break;
                case GPU_SHADER_BREV:
                    for (uint32_t bit = 0; bit < 32U; ++bit)
                    {
                        value |= ((first >> bit) & 1U) << (31U - bit);
                    }
                    break;
                case GPU_SHADER_IABS:
                    value = 0 != (first & 0x80000000U) ? 0U - first : first;
                    break;
                case GPU_SHADER_SHF:
                    value = funnel_shift(first, second, third, &instruction);
                    break;
                case GPU_SHADER_FLO:
                    value = find_leading(first, &instruction);
                    break;
                case GPU_SHADER_PRMT:
                    value = permute_bytes(first, second, third, instruction.permute_mode);
                    break;
                case GPU_SHADER_SGXT:
                    value = extract_low_bits(first,
                                             0 != instruction.shift_wrap ? second & 31U : second,
                                             instruction.compare_signed);
                    break;
                case GPU_SHADER_BMSK: {
                    const uint32_t position = 0 != instruction.shift_wrap ? first & 31U : first;
                    const uint32_t width    = 0 != instruction.shift_wrap ? second & 31U : second;
                    const uint32_t mask     = width >= 32U ? UINT32_MAX : (1U << width) - 1U;
                    value                   = position >= 32U ? 0 : mask << position;
                    break;
                }
                case GPU_SHADER_IMNMX: {
                    const uint32_t sign    = 0 != instruction.compare_signed ? 0x80000000U : 0;
                    const uint32_t less    = (first ^ sign) < (second ^ sign);
                    const uint32_t minimum = predicate_value(
                        lane, instruction.condition, instruction.condition_invert);
                    value = less == minimum ? first : second;
                    break;
                }
                case GPU_SHADER_SEL:
                    value = 0
                                    != predicate_value(
                                        lane, instruction.condition, instruction.condition_invert)
                                ? first
                                : second;
                    break;
                case GPU_SHADER_ISETP: {
                    const uint32_t comparison = compare_integer(first, second, &instruction);
                    const uint32_t condition
                        = predicate_value(lane,
                                          instruction.predicate_sources[0],
                                          instruction.predicate_source_inverts[0]);
                    predicate_results[0]
                        = combine_predicates(comparison, condition, instruction.boolean_operation);
                    predicate_results[1] = combine_predicates(
                        comparison ^ 1U, condition, instruction.boolean_operation);
                    break;
                }
                case GPU_SHADER_PLOP3: {
                    uint32_t        index  = 0;
                    const uint32_t *source = instruction.predicate_sources;
                    const uint32_t *invert = instruction.predicate_source_inverts;
                    for (uint32_t source_index = 0; source_index < 3U;
                         ++source_index, ++source, ++invert)
                    {
                        index = (index << 1U) | predicate_value(lane, *source, *invert);
                    }
                    predicate_results[0] = (instruction.predicate_lookups[0] >> index) & 1U;
                    predicate_results[1] = (instruction.predicate_lookups[1] >> index) & 1U;
                    break;
                }
                case GPU_SHADER_BRA:
                    if (0
                        != predicate_value(
                            lane, instruction.condition, instruction.condition_invert))
                    {
                        if (instruction.branch_bytes < 0)
                        {
                            const uint64_t magnitude = (uint64_t)-instruction.branch_bytes;
                            if (magnitude > next)
                            {
                                return POUND_ERROR_GPU_INVALID_SHADER;
                            }
                            next -= (size_t)magnitude;
                        }
                        else
                        {
                            const uint64_t distance = (uint64_t)instruction.branch_bytes;
                            if (distance > size - next)
                            {
                                return POUND_ERROR_GPU_INVALID_SHADER;
                            }
                            next += (size_t)distance;
                        }
                        if (next >= size)
                        {
                            return POUND_ERROR_GPU_INVALID_SHADER;
                        }
                    }
                    break;
                case GPU_SHADER_EXIT:
                    halted = 1U;
                    break;
                case GPU_SHADER_NOP:
                    break;
                default:
                    return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            uint32_t        predicates  = lane->predicates;
            const uint32_t *destination = instruction.predicate_destinations;
            const uint32_t *result      = predicate_results;
            for (uint32_t destination_index = 0;
                 destination_index < instruction.predicate_destination_count;
                 ++destination_index, ++destination, ++result)
            {
                if (GPU_SHADER_TRUE_PREDICATE != *destination)
                {
                    predicates = (predicates & ~(1U << *destination)) | (*result << *destination);
                }
            }
            lane->predicates = predicates;
            if (0 != instruction.source_count
                && GPU_SHADER_ZERO_REGISTER != instruction.destination)
            {
                lane->registers[instruction.destination] = value;
            }
        }
        pc = next;
        ++steps;
        lane->pc     = pc;
        lane->steps  = steps;
        lane->halted = halted;
    }
    return 0 != halted ? POUND_SUCCESS : POUND_ERROR_GPU_RESOURCE_LIMIT;
}
