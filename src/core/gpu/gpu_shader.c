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
    if (0x107U == operation || 0x104U == operation)
    {
        const uint32_t format = third & ~0x0001C000U;
        if ((1U != form && 4U != form) || (first & 0xFF000000U) != 0
            || (0x104U == operation && format == 0x00101000U && form != 1U)
            || (1U == form && (second & ~0xC00000FFU) != 0)
            || (0x104U == operation ? format != 0x00200800U && format != 0x00101000U
                                    : format != 0x00201000U))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        decoded.operation   = 0x107U == operation ? GPU_SHADER_FRND : GPU_SHADER_F2F;
        decoded.float_widen = 0x104U == operation && format == 0x00101000U;
        decoded.sources[0]
            = 1U == form ? register_source(second & 255U, second >> 31U) : immediate_source(second);
        decoded.sources[0].absolute = 1U == form ? (second >> 30U) & 1U : 0;
        decoded.source_count        = 1U;
        decoded.float_ftz           = (third >> 16U) & 1U;
        decoded.float_rounding      = (gpu_shader_float_rounding_t)((third >> 14U) & 3U);
        *instruction                = decoded;
        return POUND_SUCCESS;
    }
    if (0x105U == operation || 0x106U == operation || 0x045U == operation)
    {
        const uint32_t to_integer = 0x105U == operation;
        const uint32_t flags      = to_integer ? 0x0001E100U : 0x0000C400U;
        if ((1U != form && 4U != form) || (first & 0xFF000000U) != 0
            || (1U == form && (second & ~(to_integer ? 0xC00000FFU : 0x000000FFU)) != 0)
            || (third & ~flags) != 0x00201000U
            || (0x045U == operation && (third & 0x0000C000U) != 0
                && (third & 0x0000C000U) != 0x0000C000U))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        decoded.operation           = to_integer ? GPU_SHADER_F2I
                                                 : (0x045U == operation ? GPU_SHADER_I2FP : GPU_SHADER_I2F);
        decoded.sources[0]          = 1U == form
                                          ? register_source(second & 255U, to_integer ? second >> 31U : 0)
                                          : immediate_source(second);
        decoded.sources[0].absolute = 1U == form && to_integer ? (second >> 30U) & 1U : 0;
        decoded.source_count        = 1U;
        decoded.conversion_signed   = (third >> (to_integer ? 8U : 10U)) & 1U;
        decoded.float_ftz           = to_integer ? (third >> 16U) & 1U : 0;
        decoded.float_nan_to_zero   = to_integer ? (third >> 13U) & 1U : 0;
        decoded.float_rounding      = (gpu_shader_float_rounding_t)((third >> 14U) & 3U);
        *instruction                = decoded;
        return POUND_SUCCESS;
    }
    if (0x023U == operation)
    {
        if ((1U != form && 2U != form && 4U != form) || (1U == form && 0 != (second & ~0xC00000FFU))
            || 0 != (third & ~0x0001EFFFU))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        decoded.operation              = GPU_SHADER_FFMA;
        decoded.sources[0]             = register_source(first >> 24U, (third >> 8U) & 1U);
        decoded.sources[0].absolute    = (third >> 9U) & 1U;
        const gpu_shader_source_t last = { .kind     = GPU_SHADER_SOURCE_REGISTER,
                                           .value    = third & 255U,
                                           .negate   = (third >> 11U) & 1U,
                                           .absolute = (third >> 10U) & 1U };
        decoded.sources[1]             = 2U == form
                                             ? last
                                             : (1U == form ? register_source(second & 255U, second >> 31U)
                                                           : immediate_source(second));
        if (1U == form)
        {
            decoded.sources[1].absolute = (second >> 30U) & 1U;
        }
        decoded.sources[2]     = 2U == form ? immediate_source(second) : last;
        decoded.source_count   = 3U;
        decoded.float_ftz      = (third >> 16U) & 1U;
        decoded.float_rounding = (gpu_shader_float_rounding_t)((third >> 14U) & 3U);
        decoded.float_saturate = (third >> 13U) & 1U;
        *instruction           = decoded;
        return POUND_SUCCESS;
    }
    if (0x020U == operation || 0x021U == operation)
    {
        const uint32_t immediate_form = 0x021U == operation ? 2U : 4U;
        if ((1U != form && immediate_form != form) || (1U == form && 0 != (second & ~0xC00000FFU))
            || 0 != (third & ~0x0041E300U)
            || (third & 0x00400000U) != (0x020U == operation ? 0x00400000U : 0))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        decoded.operation           = 0x020U == operation ? GPU_SHADER_FMUL : GPU_SHADER_FADD;
        decoded.sources[0]          = register_source(first >> 24U, (third >> 8U) & 1U);
        decoded.sources[0].absolute = (third >> 9U) & 1U;
        decoded.sources[1]
            = 1U == form ? register_source(second & 255U, second >> 31U) : immediate_source(second);
        decoded.sources[1].absolute = 1U == form ? (second >> 30U) & 1U : 0;
        decoded.source_count        = 2U;
        decoded.float_ftz           = (third >> 16U) & 1U;
        decoded.float_rounding      = (gpu_shader_float_rounding_t)((third >> 14U) & 3U);
        decoded.float_saturate      = (third >> 13U) & 1U;
        *instruction                = decoded;
        return POUND_SUCCESS;
    }
    if (0x009U == operation || 0x00BU == operation)
    {
        if ((1U != form && 4U != form) || (1U == form && 0 != (second & ~0xC00000FFU))
            || (0x009U == operation && 0 != (third & ~0x07810300U))
            || (0x00BU == operation
                && (0 != (first & 0x00FF0000U) || 0 != (third & ~0x07FFFF00U)
                    || 3U == ((third >> 10U) & 3U))))
        {
            return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
        }
        decoded.operation           = 0x009U == operation ? GPU_SHADER_FMNMX : GPU_SHADER_FSETP;
        decoded.sources[0]          = register_source(first >> 24U, (third >> 8U) & 1U);
        decoded.sources[0].absolute = (third >> 9U) & 1U;
        decoded.sources[1]
            = 1U == form ? register_source(second & 255U, second >> 31U) : immediate_source(second);
        decoded.sources[1].absolute = 1U == form ? (second >> 30U) & 1U : 0;
        decoded.source_count        = 2U;
        decoded.float_ftz           = (third >> 16U) & 1U;
        if (GPU_SHADER_FSETP == decoded.operation)
        {
            decoded.destination          = GPU_SHADER_ZERO_REGISTER;
            decoded.float_comparison     = (gpu_shader_float_comparison_t)((third >> 12U) & 15U);
            decoded.boolean_operation    = (gpu_shader_boolean_t)((third >> 10U) & 3U);
            decoded.predicate_sources[0] = decoded.condition;
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
        }
        *instruction = decoded;
        return POUND_SUCCESS;
    }
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
            decoded.operation      = GPU_SHADER_IMAD;
            decoded.source_count   = 3U;
            decoded.compare_signed = (third >> 9U) & 1U;
            break;
        case 0x025U:
        case 0x027U:
            if (0x078E0000U != (third & ~0x2FFU)
                || (255U != (third & 255U) && (0 != (third & 1U) || (third & 255U) > 252U))
                || (0x025U == operation && GPU_SHADER_ZERO_REGISTER != decoded.destination
                    && (0 != (decoded.destination & 1U) || decoded.destination > 252U)))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation      = GPU_SHADER_IMAD;
            decoded.source_count   = 3U;
            decoded.compare_signed = (third >> 9U) & 1U;
            decoded.multiply_mode
                = 0x025U == operation ? GPU_SHADER_MULTIPLY_WIDE : GPU_SHADER_MULTIPLY_HIGH;
            break;
        case 0x026U:
            if (1U != form || 0 != (third & ~0x36FFU)
                || (0 != (third & 0x2000U) && 0 == (third & 0x1000U)))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation     = 0 != (third & 0x1000U) ? GPU_SHADER_IDP2 : GPU_SHADER_IDP4;
            decoded.source_count  = 3U;
            decoded.dot_signed[0] = (third >> 9U) & 1U;
            decoded.dot_signed[1] = (third >> 10U) & 1U;
            decoded.dot_high      = (third >> 13U) & 1U;
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
            if (0 != (third & ~0x078FFFFFU))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation    = GPU_SHADER_LOP3;
            decoded.lookup       = (third >> 8U) & 255U;
            decoded.source_count = 3U;
            decoded.boolean_operation
                = 0 != (third & 0x10000U) ? GPU_SHADER_BOOLEAN_AND : GPU_SHADER_BOOLEAN_OR;
            decoded.predicate_destinations[0]   = (third >> 17U) & 7U;
            decoded.predicate_destination_count = 1U;
            decoded.predicate_sources[0]        = decoded.condition;
            decoded.predicate_source_inverts[0] = decoded.condition_invert;
            decoded.predicate_source_count      = 1U;
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
            if (0 != (first & 0x00FF0000U) || 0 != (third & ~0x07FE7FF0U)
                || (0 == (third & 0x100U) && 0x70U != (third & 0x70U))
                || 3U == ((third >> 10U) & 3U))
            {
                return POUND_ERROR_GPU_UNSUPPORTED_SHADER;
            }
            decoded.operation                   = GPU_SHADER_ISETP;
            decoded.destination                 = GPU_SHADER_ZERO_REGISTER;
            decoded.comparison                  = (gpu_shader_comparison_t)((third >> 12U) & 7U);
            decoded.boolean_operation           = (gpu_shader_boolean_t)((third >> 10U) & 3U);
            decoded.compare_signed              = (third >> 9U) & 1U;
            decoded.compare_extended            = (third >> 8U) & 1U;
            decoded.source_count                = 2U;
            decoded.predicate_sources[0]        = decoded.condition;
            decoded.predicate_source_inverts[0] = decoded.condition_invert;
            decoded.predicate_source_count      = 1U;
            if (0 != decoded.compare_extended)
            {
                decoded.predicate_sources[1]        = (third >> 4U) & 7U;
                decoded.predicate_source_inverts[1] = (third >> 7U) & 1U;
                decoded.predicate_source_count      = 2U;
            }
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
        case GPU_SHADER_IDP4:
            return "IDP4";
        case GPU_SHADER_IDP2:
            return "IDP2";
        case GPU_SHADER_FSETP:
            return "FSETP";
        case GPU_SHADER_FMNMX:
            return "FMNMX";
        case GPU_SHADER_FADD:
            return "FADD";
        case GPU_SHADER_FMUL:
            return "FMUL";
        case GPU_SHADER_FFMA:
            return "FFMA";
        case GPU_SHADER_F2I:
            return "F2I";
        case GPU_SHADER_I2F:
            return "I2F";
        case GPU_SHADER_FRND:
            return "FRND";
        case GPU_SHADER_I2FP:
            return "I2FP";
        case GPU_SHADER_F2F:
            return "F2F";
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
source_value(const gpu_shader_lane_t   *lane,
             const gpu_shader_source_t *source,
             uint32_t                   bitwise,
             uint32_t                   floating)
{
    const uint32_t value
        = GPU_SHADER_SOURCE_IMMEDIATE == source->kind
              ? source->value
              : (GPU_SHADER_ZERO_REGISTER == source->value ? 0 : lane->registers[source->value]);
    if (floating != 0)
    {
        const uint32_t absolute = source->absolute != 0 ? value & 0x7FFFFFFFU : value;
        return source->negate != 0 ? absolute ^ 0x80000000U : absolute;
    }
    return 0 == source->negate ? value : (0 != bitwise ? ~value : 0U - value);
}

static uint32_t
compare_integer(uint32_t                first,
                uint32_t                second,
                gpu_shader_comparison_t comparison,
                uint32_t                signed_value)
{
    if (0 != signed_value)
    {
        first ^= 0x80000000U;
        second ^= 0x80000000U;
    }
    switch (comparison)
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
float_flush(uint32_t value, uint32_t ftz)
{
    return ftz != 0 && (value & 0x7F800000U) == 0 ? value & 0x80000000U : value;
}

static uint32_t
float_key(uint32_t value)
{
    return (value & 0x80000000U) != 0 ? ~value : value ^ 0x80000000U;
}

static uint32_t
float_nan(uint32_t value)
{
    return (value & 0x7FFFFFFFU) > 0x7F800000U;
}

static uint32_t
compare_float(uint32_t first, uint32_t second, gpu_shader_float_comparison_t operation)
{
    const uint32_t nan      = float_nan(first) | float_nan(second);
    const uint32_t left     = (first & 0x7FFFFFFFU) == 0 ? 0 : first;
    const uint32_t right    = (second & 0x7FFFFFFFU) == 0 ? 0 : second;
    const uint32_t relation = compare_integer(
        float_key(left), float_key(right), (gpu_shader_comparison_t)(operation & 7U), 0);
    return (operation & 8U) != 0 ? relation | nan : relation & (nan ^ 1U);
}

static uint32_t
float_minmax(uint32_t first, uint32_t second, uint32_t minimum)
{
    const uint32_t left_nan  = float_nan(first);
    const uint32_t right_nan = float_nan(second);
    if (left_nan != 0 || right_nan != 0)
    {
        return left_nan != 0 ? (right_nan != 0 ? 0x7FFFFFFFU : second) : first;
    }
    return (float_key(first) < float_key(second)) == minimum ? first : second;
}

static uint32_t
float_shift_jam(uint32_t value, uint32_t distance)
{
    return distance == 0
               ? value
               : (distance >= 32U ? value != 0
                                  : (value >> distance) | ((value & ((1U << distance) - 1U)) != 0));
}

static uint32_t
float_round(uint32_t                    significand,
            int32_t                     exponent,
            uint32_t                    sign,
            gpu_shader_float_rounding_t rounding,
            uint32_t                    ftz)
{
    if (exponent < 1)
    {
        significand = float_shift_jam(significand, (uint32_t)(1 - exponent));
        exponent    = 1;
    }
    if (ftz && exponent == 1)
    {
        const uint32_t round_outward = (rounding == GPU_SHADER_ROUND_DOWN && sign != 0)
                                       || (rounding == GPU_SHADER_ROUND_UP && sign == 0);
        const uint32_t normal
            = rounding == GPU_SHADER_ROUND_NEAREST
                  ? significand >= 0x03FFFFFEU
                  : (round_outward ? significand > 0x03FFFFFCU : significand >= 0x04000000U);
        if (!normal)
        {
            return sign;
        }
    }
    const uint32_t discarded = significand & 7U;
    uint32_t       mantissa  = significand >> 3U;
    const uint32_t increment = rounding == GPU_SHADER_ROUND_NEAREST
                                   ? discarded > 4U || (discarded == 4U && (mantissa & 1U) != 0)
                                   : discarded != 0
                                         && ((rounding == GPU_SHADER_ROUND_DOWN && sign != 0)
                                             || (rounding == GPU_SHADER_ROUND_UP && sign == 0));
    mantissa += increment;
    if (mantissa >= 0x01000000U)
    {
        mantissa >>= 1U;
        ++exponent;
    }
    if (exponent >= 255)
    {
        const uint32_t infinity = rounding == GPU_SHADER_ROUND_NEAREST
                                  || (rounding == GPU_SHADER_ROUND_DOWN && sign != 0)
                                  || (rounding == GPU_SHADER_ROUND_UP && sign == 0);
        return sign | (infinity ? 0x7F800000U : 0x7F7FFFFFU);
    }
    return sign | (mantissa >= 0x00800000U ? (uint32_t)exponent << 23U : 0)
           | (mantissa & 0x007FFFFFU);
}

static uint32_t
half_to_float(uint32_t value, const gpu_shader_instruction_t *instruction)
{
    uint32_t sign = (value & 0x8000U) << 16U;
    if (instruction->sources[0].absolute)
    {
        sign = 0;
    }
    if (instruction->sources[0].negate)
    {
        sign ^= 0x80000000U;
    }
    uint32_t exponent = (value >> 10U) & 31U;
    uint32_t fraction = value & 0x3FFU;
    if (exponent == 31U)
    {
        return fraction ? 0x7FFFFFFFU : sign | 0x7F800000U;
    }
    if (exponent == 0)
    {
        if (fraction == 0 || instruction->float_ftz)
        {
            return sign;
        }
        exponent = 113U;
        while (fraction < 0x400U)
        {
            fraction <<= 1U;
            --exponent;
        }
    }
    else
    {
        exponent += 112U;
    }
    return sign | (exponent << 23U) | ((fraction & 0x3FFU) << 13U);
}

static uint32_t
float_to_half(uint32_t value, const gpu_shader_instruction_t *instruction)
{
    value                    = float_flush(value, instruction->float_ftz);
    const uint32_t sign      = (value >> 16U) & 0x8000U;
    const uint32_t magnitude = value & 0x7FFFFFFFU;
    if (magnitude >= 0x7F800000U)
    {
        return magnitude > 0x7F800000U ? 0x7FFFU : sign | 0x7C00U;
    }
    const uint32_t source_exponent = (value >> 23U) & 255U;
    int32_t        exponent        = (int32_t)(source_exponent ? source_exponent : 1U) - 112;
    const uint32_t significand     = (value & 0x007FFFFFU) | (source_exponent ? 0x00800000U : 0);
    const uint32_t distance        = exponent < 1 ? (uint32_t)(14 - exponent) : 13U;
    const uint32_t discarded
        = distance >= 32U ? significand : significand & ((1U << distance) - 1U);
    uint32_t       mantissa = distance >= 32U ? 0 : significand >> distance;
    const uint32_t half     = distance >= 32U ? UINT32_MAX : 1U << (distance - 1U);
    const uint32_t increment
        = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST
              ? discarded > half || (discarded == half && (mantissa & 1U) != 0)
              : discarded != 0
                    && ((instruction->float_rounding == GPU_SHADER_ROUND_DOWN && sign != 0)
                        || (instruction->float_rounding == GPU_SHADER_ROUND_UP && sign == 0));
    mantissa += increment;
    exponent = exponent < 1 ? 1 : exponent;
    if (mantissa >= 0x800U)
    {
        mantissa >>= 1U;
        ++exponent;
    }
    if (exponent >= 31)
    {
        const uint32_t infinity
            = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST
              || (instruction->float_rounding == GPU_SHADER_ROUND_DOWN && sign != 0)
              || (instruction->float_rounding == GPU_SHADER_ROUND_UP && sign == 0);
        return sign | (infinity ? 0x7C00U : 0x7BFFU);
    }
    if (instruction->float_ftz && mantissa < 0x400U)
    {
        return sign;
    }
    return sign | (mantissa >= 0x400U ? (uint32_t)exponent << 10U : 0) | (mantissa & 0x3FFU);
}

static uint32_t
float_round_integer(uint32_t value, const gpu_shader_instruction_t *instruction)
{
    value = float_flush(value, instruction->float_ftz);
    if (float_nan(value))
    {
        return 0x7FFFFFFFU;
    }
    const uint32_t sign      = value & 0x80000000U;
    const uint32_t magnitude = value & 0x7FFFFFFFU;
    const uint32_t exponent  = (value >> 23U) & 255U;
    const uint32_t outward   = (instruction->float_rounding == GPU_SHADER_ROUND_DOWN && sign != 0)
                             || (instruction->float_rounding == GPU_SHADER_ROUND_UP && sign == 0);
    if (exponent >= 150U)
    {
        return value;
    }
    if (exponent < 127U)
    {
        const uint32_t increment = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST
                                       ? magnitude > 0x3F000000U
                                       : outward && magnitude != 0;
        return sign | (increment ? 0x3F800000U : 0);
    }
    const uint32_t distance  = 150U - exponent;
    const uint32_t unit      = 1U << distance;
    const uint32_t discarded = value & (unit - 1U);
    const uint32_t integral  = value & ~(unit - 1U);
    const uint32_t half      = unit >> 1U;
    const uint32_t increment
        = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST
              ? discarded > half || (discarded == half && (integral & unit) != 0)
              : outward && discarded != 0;
    return integral + (increment ? unit : 0);
}

static uint32_t
float_to_integer(uint32_t value, const gpu_shader_instruction_t *instruction)
{
    value = float_flush(value, instruction->float_ftz);
    if (float_nan(value))
    {
        return instruction->float_nan_to_zero ? 0 : 0x80000000U;
    }
    const uint32_t negative = value >> 31U;
    const uint32_t exponent = (value >> 23U) & 255U;
    const uint32_t limit = instruction->conversion_signed ? (negative ? 0x80000000U : 0x7FFFFFFFU)
                                                          : (negative ? 0 : UINT32_MAX);
    if (exponent >= 159U)
    {
        return limit;
    }
    const uint32_t significand = (value & 0x007FFFFFU) | (exponent ? 0x00800000U : 0);
    uint32_t       magnitude;
    if (exponent >= 150U)
    {
        magnitude = significand << (exponent - 150U);
    }
    else
    {
        const uint32_t distance = 150U - exponent;
        const uint32_t discarded
            = distance >= 32U ? significand : significand & ((1U << distance) - 1U);
        magnitude           = distance >= 32U ? 0 : significand >> distance;
        const uint32_t half = distance >= 32U ? UINT32_MAX : 1U << (distance - 1U);
        const uint32_t increment
            = instruction->float_rounding == GPU_SHADER_ROUND_NEAREST
                  ? discarded > half || (discarded == half && (magnitude & 1U))
                  : discarded != 0
                        && ((instruction->float_rounding == GPU_SHADER_ROUND_DOWN && negative)
                            || (instruction->float_rounding == GPU_SHADER_ROUND_UP && !negative));
        magnitude += increment;
    }
    if (instruction->conversion_signed || negative)
    {
        if (magnitude > limit)
        {
            magnitude = limit;
        }
    }
    return negative ? 0U - magnitude : magnitude;
}

static uint32_t
integer_to_float(uint32_t value, const gpu_shader_instruction_t *instruction)
{
    const uint32_t sign      = instruction->conversion_signed ? value & 0x80000000U : 0;
    uint32_t       magnitude = sign ? 0U - value : value;
    if (!magnitude)
    {
        return 0;
    }
    uint32_t leading = 0;
    for (uint32_t remaining = magnitude; remaining > 1U; remaining >>= 1U)
    {
        ++leading;
    }
    magnitude
        = leading > 26U ? float_shift_jam(magnitude, leading - 26U) : magnitude << (26U - leading);
    return float_round(magnitude, (int32_t)(127U + leading), sign, instruction->float_rounding, 0);
}

static uint32_t
float_add(uint32_t first, uint32_t second, gpu_shader_float_rounding_t rounding, uint32_t ftz)
{
    if (float_nan(first) || float_nan(second))
    {
        return 0x7FFFFFFFU;
    }
    if ((first & 0x7FFFFFFFU) < (second & 0x7FFFFFFFU))
    {
        const uint32_t temporary = first;
        first                    = second;
        second                   = temporary;
    }
    const uint32_t sign             = first & 0x80000000U;
    const uint32_t matching_signs   = ((first ^ second) & 0x80000000U) == 0;
    int32_t        exponent         = (int32_t)((first >> 23U) & 255U);
    int32_t        smaller_exponent = (int32_t)((second >> 23U) & 255U);
    if (exponent == 255)
    {
        return smaller_exponent == 255 && !matching_signs ? 0x7FFFFFFFU : first;
    }
    uint32_t larger  = ((first & 0x007FFFFFU) | (exponent != 0 ? 0x00800000U : 0)) << 3U;
    uint32_t smaller = ((second & 0x007FFFFFU) | (smaller_exponent != 0 ? 0x00800000U : 0)) << 3U;
    exponent         = exponent == 0 ? 1 : exponent;
    smaller_exponent = smaller_exponent == 0 ? 1 : smaller_exponent;
    smaller          = float_shift_jam(smaller, (uint32_t)(exponent - smaller_exponent));
    uint32_t significand = matching_signs ? larger + smaller : larger - smaller;
    if (significand == 0)
    {
        return matching_signs ? sign : (rounding == GPU_SHADER_ROUND_DOWN ? 0x80000000U : 0);
    }
    if (significand >= 0x08000000U)
    {
        significand = float_shift_jam(significand, 1U);
        ++exponent;
    }
    while (significand < 0x04000000U && exponent > 1)
    {
        significand <<= 1U;
        --exponent;
    }
    return float_round(significand, exponent, sign, rounding, ftz);
}

static uint32_t
float_multiply(uint32_t first, uint32_t second, gpu_shader_float_rounding_t rounding, uint32_t ftz)
{
    const uint32_t sign  = (first ^ second) & 0x80000000U;
    const uint32_t left  = first & 0x7FFFFFFFU;
    const uint32_t right = second & 0x7FFFFFFFU;
    if (float_nan(first) || float_nan(second) || (left == 0x7F800000U && right == 0)
        || (right == 0x7F800000U && left == 0))
    {
        return 0x7FFFFFFFU;
    }
    if (left == 0x7F800000U || right == 0x7F800000U)
    {
        return sign | 0x7F800000U;
    }
    if (left == 0 || right == 0)
    {
        return sign;
    }
    int32_t  left_exponent  = (int32_t)(left >> 23U);
    int32_t  right_exponent = (int32_t)(right >> 23U);
    uint32_t left_mantissa  = (left & 0x007FFFFFU) | (left_exponent != 0 ? 0x00800000U : 0);
    uint32_t right_mantissa = (right & 0x007FFFFFU) | (right_exponent != 0 ? 0x00800000U : 0);
    left_exponent           = left_exponent == 0 ? 1 : left_exponent;
    right_exponent          = right_exponent == 0 ? 1 : right_exponent;
    while (left_mantissa < 0x00800000U)
    {
        left_mantissa <<= 1U;
        --left_exponent;
    }
    while (right_mantissa < 0x00800000U)
    {
        right_mantissa <<= 1U;
        --right_exponent;
    }
    const uint64_t product  = (uint64_t)left_mantissa * right_mantissa;
    const uint32_t high     = (uint32_t)(product >> 47U);
    const uint32_t distance = 20U + high;
    const uint32_t significand
        = (uint32_t)(product >> distance) | ((product & ((UINT64_C(1) << distance) - 1U)) != 0);
    return float_round(
        significand, left_exponent + right_exponent - 127 + (int32_t)high, sign, rounding, ftz);
}

static uint32_t
float_arithmetic(uint32_t first, uint32_t second, const gpu_shader_instruction_t *instruction)
{
    first  = float_flush(first, instruction->float_ftz);
    second = float_flush(second, instruction->float_ftz);
    uint32_t value
        = instruction->operation == GPU_SHADER_FADD
              ? float_add(first, second, instruction->float_rounding, instruction->float_ftz)
              : float_multiply(first, second, instruction->float_rounding, instruction->float_ftz);
    value = float_flush(value, instruction->float_ftz);
    if (instruction->float_saturate)
    {
        value = (value & 0x80000000U) != 0 || float_nan(value)
                    ? 0
                    : (value > 0x3F800000U ? 0x3F800000U : value);
    }
    return value;
}

static uint32_t
float_normalized_mantissa(uint32_t value, int32_t *exponent)
{
    *exponent         = (int32_t)((value >> 23U) & 255U);
    uint32_t mantissa = (value & 0x007FFFFFU) | (*exponent != 0 ? 0x00800000U : 0);
    *exponent         = *exponent == 0 ? 1 : *exponent;
    while (mantissa != 0 && mantissa < 0x00800000U)
    {
        mantissa <<= 1U;
        --*exponent;
    }
    return mantissa;
}

static uint64_t
float_wide_shift_jam(uint64_t value, uint32_t distance)
{
    return distance == 0
               ? value
               : (distance >= 64U
                      ? value != 0
                      : (value >> distance) | ((value & ((UINT64_C(1) << distance) - 1U)) != 0));
}

static uint32_t
float_fused(uint32_t                        first,
            uint32_t                        second,
            uint32_t                        third,
            const gpu_shader_instruction_t *instruction)
{
    first                      = float_flush(first, instruction->float_ftz);
    second                     = float_flush(second, instruction->float_ftz);
    third                      = float_flush(third, instruction->float_ftz);
    const uint32_t left        = first & 0x7FFFFFFFU;
    const uint32_t right       = second & 0x7FFFFFFFU;
    const uint32_t addend      = third & 0x7FFFFFFFU;
    uint32_t       sign        = (first ^ second) & 0x80000000U;
    const uint32_t addend_sign = third & 0x80000000U;
    uint32_t       value;
    if (float_nan(first) || float_nan(second) || float_nan(third)
        || (left == 0x7F800000U && right == 0) || (right == 0x7F800000U && left == 0))
    {
        value = 0x7FFFFFFFU;
    }
    else if (left == 0x7F800000U || right == 0x7F800000U)
    {
        value = addend == 0x7F800000U && sign != addend_sign ? 0x7FFFFFFFU : sign | 0x7F800000U;
    }
    else if (addend == 0x7F800000U)
    {
        value = third;
    }
    else if ((left == 0 || right == 0) && addend == 0)
    {
        value = sign == addend_sign
                    ? sign
                    : (instruction->float_rounding == GPU_SHADER_ROUND_DOWN ? 0x80000000U : 0);
    }
    else if (left == 0 || right == 0)
    {
        value = third;
    }
    else if (addend == 0)
    {
        value = float_multiply(first, second, instruction->float_rounding, instruction->float_ftz);
    }
    else
    {
        int32_t        left_exponent, right_exponent, addend_exponent;
        const uint32_t left_mantissa   = float_normalized_mantissa(first, &left_exponent);
        const uint32_t right_mantissa  = float_normalized_mantissa(second, &right_exponent);
        const uint32_t addend_mantissa = float_normalized_mantissa(third, &addend_exponent);
        const uint64_t product         = (uint64_t)left_mantissa * right_mantissa;
        const uint32_t high            = (uint32_t)(product >> 47U);
        int32_t        exponent        = left_exponent + right_exponent - 127 + (int32_t)high;
        uint64_t       larger          = product << (16U - high);
        uint64_t       smaller         = (uint64_t)addend_mantissa << 39U;
        const uint32_t matching_signs  = sign == addend_sign;
        if (exponent < addend_exponent || (exponent == addend_exponent && larger < smaller))
        {
            const uint64_t temporary         = larger;
            larger                           = smaller;
            smaller                          = temporary;
            const int32_t temporary_exponent = exponent;
            exponent                         = addend_exponent;
            addend_exponent                  = temporary_exponent;
            sign                             = addend_sign;
        }
        smaller = float_wide_shift_jam(smaller, (uint32_t)(exponent - addend_exponent));
        uint64_t significand = matching_signs ? larger + smaller : larger - smaller;
        if (significand == 0)
        {
            value = instruction->float_rounding == GPU_SHADER_ROUND_DOWN ? 0x80000000U : 0;
        }
        else
        {
            if ((significand & (UINT64_C(1) << 63U)) != 0)
            {
                significand = float_wide_shift_jam(significand, 1U);
                ++exponent;
            }
            while ((significand & (UINT64_C(1) << 62U)) == 0)
            {
                significand <<= 1U;
                --exponent;
            }
            value = float_round((uint32_t)float_wide_shift_jam(significand, 36U),
                                exponent,
                                sign,
                                instruction->float_rounding,
                                instruction->float_ftz);
        }
    }
    value = float_flush(value, instruction->float_ftz);
    if (instruction->float_saturate)
    {
        value = (value & 0x80000000U) != 0 || float_nan(value)
                    ? 0
                    : (value > 0x3F800000U ? 0x3F800000U : value);
    }
    return value;
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

static uint32_t
packed_dot(uint32_t                        first,
           uint32_t                        second,
           uint32_t                        accumulator,
           const gpu_shader_instruction_t *instruction)
{
    const uint32_t two     = GPU_SHADER_IDP2 == instruction->operation;
    const uint32_t count   = 0 != two ? 2U : 4U;
    const uint32_t width   = 0 != two ? 16U : 8U;
    const uint32_t modulus = 1U << width;
    uint32_t       result  = accumulator;
    for (uint32_t element = 0; element < count; ++element)
    {
        const uint32_t first_bits = (first >> (width * element)) & (modulus - 1U);
        const uint32_t second_bits
            = (second >> (8U * (element + 2U * instruction->dot_high))) & 255U;
        const int32_t first_value
            = (int32_t)first_bits
              - (0 != instruction->dot_signed[0] && first_bits >= modulus / 2U ? (int32_t)modulus
                                                                               : 0);
        const int32_t second_value
            = (int32_t)second_bits
              - (0 != instruction->dot_signed[1] && second_bits >= 128U ? 256 : 0);
        result += (uint32_t)(first_value * second_value);
    }
    return result;
}

static uint64_t
multiply_integer(uint32_t first, uint32_t second, uint32_t signed_value)
{
    if (0 == signed_value)
    {
        return (uint64_t)first * second;
    }
    const int64_t first_value
        = (int64_t)first - (0 != (first & 0x80000000U) ? INT64_C(0x100000000) : 0);
    const int64_t second_value
        = (int64_t)second - (0 != (second & 0x80000000U) ? INT64_C(0x100000000) : 0);
    return (uint64_t)(first_value * second_value);
}

static error_t
execute_program(gpu_shader_lane_t       *lane,
                const void              *program,
                size_t                   size,
                uint32_t                 budget,
                const gpu_shader_node_t *nodes)
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
        gpu_shader_instruction_t        decoded;
        const gpu_shader_instruction_t *instruction;
        if (NULL != nodes)
        {
            instruction = &nodes[pc / GPU_SHADER_INSTRUCTION_SIZE].instruction;
        }
        else
        {
            const error_t status
                = gpu_shader_decode(bytes + pc, GPU_SHADER_INSTRUCTION_SIZE, &decoded);
            if (POUND_SUCCESS != status)
            {
                return status;
            }
            instruction = &decoded;
        }
        size_t next = pc + GPU_SHADER_INSTRUCTION_SIZE;
        if (0 != predicate_value(lane, instruction->predicate, instruction->predicate_invert))
        {
            uint32_t       value                = 0;
            uint32_t       predicate_results[2] = { 0 };
            uint64_t       wide_result          = 0;
            const uint32_t complement           = GPU_SHADER_POPC == instruction->operation
                                        || GPU_SHADER_FLO == instruction->operation
                                        || ((GPU_SHADER_IADD3 == instruction->operation
                                             || GPU_SHADER_LEA == instruction->operation)
                                            && 0 != instruction->carry_in);
            const uint32_t floating = instruction->operation == GPU_SHADER_FSETP
                                      || instruction->operation == GPU_SHADER_FMNMX
                                      || instruction->operation == GPU_SHADER_FADD
                                      || instruction->operation == GPU_SHADER_FMUL
                                      || instruction->operation == GPU_SHADER_FFMA
                                      || instruction->operation == GPU_SHADER_F2I
                                      || instruction->operation == GPU_SHADER_FRND
                                      || instruction->operation == GPU_SHADER_F2F;
            const uint32_t first
                = source_value(lane, &instruction->sources[0], complement, floating);
            const uint32_t second
                = source_value(lane, &instruction->sources[1], complement, floating);
            const uint32_t third
                = source_value(lane, &instruction->sources[2], complement, floating);
            switch (instruction->operation)
            {
                case GPU_SHADER_MOV:
                    value = first;
                    break;
                case GPU_SHADER_IADD3: {
                    uint64_t sum = (uint64_t)first + second + third;
                    if (0 != instruction->carry_in)
                    {
                        for (uint32_t index = 0; index < 2U; ++index)
                        {
                            sum += predicate_value(lane,
                                                   instruction->predicate_sources[index],
                                                   instruction->predicate_source_inverts[index]);
                        }
                    }
                    else
                    {
                        const uint32_t             values[3] = { first, second, third };
                        const uint32_t            *operand   = values;
                        const gpu_shader_source_t *source    = instruction->sources;
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
                    if (GPU_SHADER_MULTIPLY_LOW == instruction->multiply_mode)
                    {
                        value = first * second + third;
                    }
                    else
                    {
                        const uint32_t accumulator = instruction->sources[2].value;
                        const uint32_t upper       = GPU_SHADER_ZERO_REGISTER == accumulator
                                                         ? 0
                                                         : lane->registers[accumulator + 1U];
                        wide_result = multiply_integer(first, second, instruction->compare_signed)
                                      + ((uint64_t)third | ((uint64_t)upper << 32U));
                        value = (uint32_t)(GPU_SHADER_MULTIPLY_HIGH == instruction->multiply_mode
                                               ? wide_result >> 32U
                                               : wide_result);
                    }
                    break;
                case GPU_SHADER_IDP4:
                case GPU_SHADER_IDP2:
                    value = packed_dot(first, second, third, instruction);
                    break;
                case GPU_SHADER_LEA: {
                    const uint32_t shifted
                        = 0 != instruction->shift_high
                              ? (uint32_t)(((uint64_t)first | ((uint64_t)third << 32U))
                                           >> (32U - instruction->address_shift))
                              : first << instruction->address_shift;
                    uint64_t sum = shifted;
                    if (0 != instruction->intermediate_negate)
                    {
                        sum = (uint64_t)~shifted + (0 != instruction->carry_in ? 0U : 1U);
                    }
                    sum += second;
                    if (0 != instruction->carry_in)
                    {
                        sum += predicate_value(
                            lane, instruction->condition, instruction->condition_invert);
                    }
                    else if (0 != instruction->sources[1].negate && 0 == second)
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
                        value |= ((instruction->lookup >> index) & 1U) << bit;
                    }
                    predicate_results[0] = combine_predicates(
                        0 != value,
                        predicate_value(lane,
                                        instruction->predicate_sources[0],
                                        instruction->predicate_source_inverts[0]),
                        instruction->boolean_operation);
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
                    value = funnel_shift(first, second, third, instruction);
                    break;
                case GPU_SHADER_FLO:
                    value = find_leading(first, instruction);
                    break;
                case GPU_SHADER_PRMT:
                    value = permute_bytes(first, second, third, instruction->permute_mode);
                    break;
                case GPU_SHADER_SGXT:
                    value = extract_low_bits(first,
                                             0 != instruction->shift_wrap ? second & 31U : second,
                                             instruction->compare_signed);
                    break;
                case GPU_SHADER_BMSK: {
                    const uint32_t position = 0 != instruction->shift_wrap ? first & 31U : first;
                    const uint32_t width    = 0 != instruction->shift_wrap ? second & 31U : second;
                    const uint32_t mask     = width >= 32U ? UINT32_MAX : (1U << width) - 1U;
                    value                   = position >= 32U ? 0 : mask << position;
                    break;
                }
                case GPU_SHADER_IMNMX: {
                    const uint32_t sign    = 0 != instruction->compare_signed ? 0x80000000U : 0;
                    const uint32_t less    = (first ^ sign) < (second ^ sign);
                    const uint32_t minimum = predicate_value(
                        lane, instruction->condition, instruction->condition_invert);
                    value = less == minimum ? first : second;
                    break;
                }
                case GPU_SHADER_FRND:
                    value = float_round_integer(first, instruction);
                    break;
                case GPU_SHADER_F2F: {
                    gpu_shader_source_t operand = instruction->sources[0];
                    operand.negate              = 0;
                    value                       = instruction->float_widen
                                                      ? half_to_float(source_value(lane, &operand, 0, 0), instruction)
                                                      : float_to_half(first, instruction);
                    break;
                }
                case GPU_SHADER_F2I:
                    value = float_to_integer(first, instruction);
                    break;
                case GPU_SHADER_I2F:
                case GPU_SHADER_I2FP:
                    value = integer_to_float(first, instruction);
                    break;
                case GPU_SHADER_FMNMX:
                    value = float_minmax(float_flush(first, instruction->float_ftz),
                                         float_flush(second, instruction->float_ftz),
                                         predicate_value(lane,
                                                         instruction->condition,
                                                         instruction->condition_invert));
                    break;
                case GPU_SHADER_FADD:
                case GPU_SHADER_FMUL:
                    value = float_arithmetic(first, second, instruction);
                    break;
                case GPU_SHADER_FFMA:
                    value = float_fused(first, second, third, instruction);
                    break;
                case GPU_SHADER_FSETP: {
                    const uint32_t comparison
                        = compare_float(float_flush(first, instruction->float_ftz),
                                        float_flush(second, instruction->float_ftz),
                                        instruction->float_comparison);
                    const uint32_t condition = predicate_value(
                        lane, instruction->condition, instruction->condition_invert);
                    predicate_results[0]
                        = combine_predicates(comparison, condition, instruction->boolean_operation);
                    predicate_results[1] = combine_predicates(
                        comparison ^ 1U, condition, instruction->boolean_operation);
                    break;
                }
                case GPU_SHADER_SEL:
                    value = 0
                                    != predicate_value(
                                        lane, instruction->condition, instruction->condition_invert)
                                ? first
                                : second;
                    break;
                case GPU_SHADER_ISETP: {
                    uint32_t comparison = compare_integer(
                        first, second, instruction->comparison, instruction->compare_signed);
                    if (0 != instruction->compare_extended && first == second
                        && GPU_SHADER_COMPARE_FALSE != instruction->comparison
                        && GPU_SHADER_COMPARE_TRUE != instruction->comparison)
                    {
                        comparison = predicate_value(lane,
                                                     instruction->predicate_sources[1],
                                                     instruction->predicate_source_inverts[1]);
                    }
                    const uint32_t condition
                        = predicate_value(lane,
                                          instruction->predicate_sources[0],
                                          instruction->predicate_source_inverts[0]);
                    predicate_results[0]
                        = combine_predicates(comparison, condition, instruction->boolean_operation);
                    predicate_results[1] = combine_predicates(
                        comparison ^ 1U, condition, instruction->boolean_operation);
                    break;
                }
                case GPU_SHADER_PLOP3: {
                    uint32_t        index  = 0;
                    const uint32_t *source = instruction->predicate_sources;
                    const uint32_t *invert = instruction->predicate_source_inverts;
                    for (uint32_t source_index = 0; source_index < 3U;
                         ++source_index, ++source, ++invert)
                    {
                        index = (index << 1U) | predicate_value(lane, *source, *invert);
                    }
                    predicate_results[0] = (instruction->predicate_lookups[0] >> index) & 1U;
                    predicate_results[1] = (instruction->predicate_lookups[1] >> index) & 1U;
                    break;
                }
                case GPU_SHADER_BRA:
                    if (0
                        != predicate_value(
                            lane, instruction->condition, instruction->condition_invert))
                    {
                        if (instruction->branch_bytes < 0)
                        {
                            const uint64_t magnitude = (uint64_t)-instruction->branch_bytes;
                            if (magnitude > next)
                            {
                                return POUND_ERROR_GPU_INVALID_SHADER;
                            }
                            next -= (size_t)magnitude;
                        }
                        else
                        {
                            const uint64_t distance = (uint64_t)instruction->branch_bytes;
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
            const uint32_t *destination = instruction->predicate_destinations;
            const uint32_t *result      = predicate_results;
            for (uint32_t destination_index = 0;
                 destination_index < instruction->predicate_destination_count;
                 ++destination_index, ++destination, ++result)
            {
                if (GPU_SHADER_TRUE_PREDICATE != *destination)
                {
                    predicates = (predicates & ~(1U << *destination)) | (*result << *destination);
                }
            }
            lane->predicates = predicates;
            if (0 != instruction->source_count
                && GPU_SHADER_ZERO_REGISTER != instruction->destination)
            {
                lane->registers[instruction->destination] = value;
                if (GPU_SHADER_IMAD == instruction->operation
                    && GPU_SHADER_MULTIPLY_WIDE == instruction->multiply_mode)
                {
                    lane->registers[instruction->destination + 1U] = (uint32_t)(wide_result >> 32U);
                }
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

error_t
gpu_shader_run(gpu_shader_lane_t *lane, const void *program, size_t size, uint32_t budget)
{
    return execute_program(lane, program, size, budget, NULL);
}

error_t
gpu_shader_run_prepared(gpu_shader_lane_t          *lane,
                        const gpu_shader_program_t *program,
                        uint32_t                    budget)
{
    if (NULL == program || NULL == program->nodes || NULL == program->blocks
        || 0 == program->instruction_count || 0 == program->block_count
        || program->block_count > program->instruction_count
        || program->instruction_count > SIZE_MAX / GPU_SHADER_INSTRUCTION_SIZE
        || program->byte_size != program->instruction_count * GPU_SHADER_INSTRUCTION_SIZE)
    {
        return POUND_ERROR_INVALID_ARGUMENT;
    }
    return execute_program(lane, program->nodes, program->byte_size, budget, program->nodes);
}
