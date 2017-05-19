/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "lrn_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include "api/CPP/data.hpp"

#include <algorithm>

using namespace cldnn;

namespace neural
{
// Kernel names.
static const std::string kernel_name = "lrn_gpu";
static const std::string kernel_name_b8 = "lrn_gpu_b8";
static const std::string kernel_name_bfyx = "lrn_gpu_bfyx";
static const std::string kernel_name_within_channel_bfyx = "lrn_gpu_within_channel_bfyx";
static const std::string kernel_name_within_channel_yxfb = "lrn_gpu_within_channel_yxfb";
template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::configurations>
{
    static constexpr neural::gpu::engine_info_internal::configurations value = neural::gpu::engine_info_internal::configurations::GT_UNKNOWN;
};


struct lrn_gpu : typed_primitive_impl<lrn>
{
    const lrn_node& outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, lrn_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    lrn_gpu(const lrn_node& arg)
        : outer(arg),
        _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(
            outer,
            outer.input().get_output_layout().data_type,
            outer.input().get_output_layout().format,
            outer.input().get_output_layout().size.batch[0],
            _engine_info.architecture,
            _engine_info.configuration)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_default(const lrn_node& arg)
    {
        auto input_layout = arg.input().get_output_layout();  // input

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = input_layout.size.batch[0] * input_layout.size.feature[0];   // B, F
        kd.gws1 = input_layout.size.spatial[0] * input_layout.size.spatial[1]; // X, Y
        kd.gws2 = 1;
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        auto& output_padding = arg.get_primitive()->output_padding;
        auto& input_padding = arg.input().get_primitive()->output_padding;

        if (arg.get_primitive()->norm_region == cldnn_lrn_norm_region_across_channel)
        {
            // TODO: add half case: b16 (b*f dividable by 128).
            if (!kd.fp16_unit_used &&                        // halfs are not used
                !input_padding &&                            // optimized kernel_batch8 does not support input padding
                input_layout.size.batch[0] % 8 == 0 && // batch_num is multiple of 8
                kd.gws0 % 64 == 0 &&
                !input_padding &&
                !output_padding)                           // batch_num * feature_num is multiple of 64
            {
                kd.gws0 /= 8;
                kd.lws0 = 8; // gws0 is dividable by 64, so after correction it will be dividable by 8.

                kd.kernel_name = kernel_name_b8;
            }
            else
            {
                kd.kernel_name = kernel_name;
            }
        }
        else if (arg.get_primitive()->norm_region == cldnn_lrn_norm_region_within_channel)
        {
            kd.kernel_name = kernel_name_within_channel_bfyx;
        }
        else
        {
            throw std::runtime_error("Invalid norm region");
        }

        // Checking for supported paddings.
        if (arg.get_output_layout().data_padding.filling_value() != 0.0f)
            throw std::runtime_error("Unknown padding mode in lrn");

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const lrn_node& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        int size = outer.get_primitive()->size;
        int pad = (size - 1) / 2;

        //alpha_div_by_size is used for norm. region: within channels, alpha is used for norm. region: across channel
        auto alpha = outer.get_primitive()->alpha;
        auto alpha_div_by_size = outer.get_primitive()->alpha / outer.get_primitive()->size;
        auto alpha_sign = std::signbit(alpha) ? -1.0f : 1.0f;
        // When used FP16 the value cannot be scaled afterwards by alpha (it must be scaled before computing sum of squares).
        auto alpha_abs_sqrt = std::sqrt(std::abs(alpha));
        auto alpha_div_by_size_abs_sqrt = std::sqrt(std::abs(alpha_div_by_size));

        auto input_padding = outer.input().get_output_layout().data_padding;
        auto input_size = outer.input().get_output_layout().size;

        auto output_padding = outer.get_primitive()->output_padding;
        auto output_size = outer.get_output_layout().size;

        int count = input_size.sizes()[0] * input_size.sizes()[1] * input_size.sizes()[2] * input_size.sizes()[3];

        gpu::jit_constants mem_consts {
            gpu::make_jit_constant("INPUT",                         input_size),
            gpu::make_jit_constant("COUNT",                         count),
            gpu::make_jit_constant("OUTPUT",                        output_size),
            gpu::make_jit_constant("P_SIZE",                        size),
            gpu::make_jit_constant("PAD",                           pad),
            gpu::make_jit_constant("ALPHA",                         data.fp16_unit_used ? alpha_sign : alpha),
            gpu::make_jit_constant("ALPHA_DIV_BY_SIZE",             data.fp16_unit_used ? alpha_sign : alpha_div_by_size),
            gpu::make_jit_constant("ALPHA_VAL_FACTOR",              data.fp16_unit_used ? alpha_abs_sqrt : 1.0f),
            gpu::make_jit_constant("ALPHA_VAL_FACTOR_DIV_BY_SIZE",  data.fp16_unit_used ? alpha_div_by_size_abs_sqrt : 1.0f),
            gpu::make_jit_constant("BETA",                          outer.get_primitive()->beta),
            gpu::make_jit_constant("K",                             outer.get_primitive()->k),
            gpu::make_jit_constant("HELP_INPUT_OFFSET",             input_padding.lower_size().negate().feature[0] - static_cast<int32_t>(size / 2)),
            gpu::make_jit_constant("FP16_SUPPORTED",                static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",                static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                     data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",                 data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("INPUT_PADDING",                 input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",                outer.get_output_layout().data_padding),
            gpu::make_jit_constant("INPUT_BUFFER_SIZE_X",           !input_padding ? input_size.spatial[0] : input_size.spatial[0] + input_padding.upper_size().spatial[0] + input_padding.lower_size().spatial[0]),
            gpu::make_jit_constant("INPUT_BUFFER_SIZE_Y",           !input_padding ? input_size.spatial[1] : input_size.spatial[1] + input_padding.upper_size().spatial[1] + input_padding.lower_size().spatial[1]),
            gpu::make_jit_constant("OUTPUT_BUFFER_SIZE_X",          !output_padding ? output_size.spatial[0] : output_size.spatial[0] + output_padding.upper_size().spatial[0] + output_padding.lower_size().spatial[0]),
            gpu::make_jit_constant("OUTPUT_BUFFER_SIZE_Y",          !output_padding ? output_size.spatial[1] : output_size.spatial[1] + output_padding.upper_size().spatial[1] + output_padding.lower_size().spatial[1]),
        };

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, lrn_inst& instance) override
    {
        const auto& kd    = _kernel_data;

        return _kernel.run<gpu::input_mem, gpu::output_mem>
            ({{ kd.gws0, kd.gws1, kd.gws2 }, { kd.lws0, kd.lws1, kd.lws2 }},
                events,
                instance.input_memory(),
                instance.output_memory());
    }


    static primitive_impl* create(const lrn_node& arg) { return new lrn_gpu(arg); }

};

lrn_gpu::kernel_data default_yxfb(const lrn_node& arg)
{
    if (arg.get_primitive()->norm_region == cldnn_lrn_norm_region_within_channel)
    {
        throw std::runtime_error("LRN within channel is not implemented for YXFB format");
    }

    lrn_gpu::kernel_data kd = lrn_gpu::set_default(arg);
    return kd;
}

lrn_gpu::kernel_data get_kernel_within_channel(const lrn_node& arg, format format)
{
    lrn_gpu::kernel_data kd = lrn_gpu::set_default(arg);

    switch (format) 
    {
        case cldnn::format::bfyx:
            kd.kernel_name = kernel_name_within_channel_bfyx;
            break;
        case cldnn::format::yxfb:
            kd.kernel_name = kernel_name_within_channel_yxfb;
            break;
        default:
            throw std::invalid_argument("Unsupported LRN within channel format - " + cldnn::format::traits(format).order);
    }    
   
    kd.gws0 = 128 * 128;
    kd.gws1 = 1;
    kd.gws2 = 1;

    kd.lws0 = 128;
    kd.lws1 = 1;
    kd.lws2 = 1;
    return kd;
}

lrn_gpu::kernel_data get_kernel_across_channel(const lrn_node& arg, format format)
{
    auto input_layout = arg.input().get_output_layout();
    lrn_gpu::kernel_data kd = lrn_gpu::set_default(arg);

    if (format == cldnn::format::bfyx)
    {
        kd.kernel_name = kernel_name_bfyx;   
        kd.gws0 = align_to(input_layout.size.spatial[0],32);
        kd.gws1 = input_layout.size.spatial[1];
        kd.gws2 = input_layout.size.feature[0] * input_layout.size.batch[0];

        kd.lws0 = 32;
        kd.lws1 = 1;
        kd.lws2 = 1;        
    }    
   
    return kd;
}


lrn_gpu::kernel_data get_yxfb_lrn_kernel(const lrn_node& arg)
{
    switch (arg.get_primitive()->norm_region)
    {
        case cldnn_lrn_norm_region_across_channel: 
            return get_kernel_across_channel(arg, cldnn::format::yxfb);
        case cldnn_lrn_norm_region_within_channel:
            return get_kernel_within_channel(arg, cldnn::format::yxfb);
        default:
            throw std::runtime_error("Invalid norm region");
    }
}


lrn_gpu::kernel_data get_bfyx_lrn_kernel(const lrn_node& arg)
{
    switch (arg.get_primitive()->norm_region)
    {
        case cldnn_lrn_norm_region_across_channel: 
            return get_kernel_across_channel(arg, cldnn::format::bfyx);
        case cldnn_lrn_norm_region_within_channel:
            return get_kernel_within_channel(arg, cldnn::format::bfyx);
        default:
            throw std::runtime_error("Invalid norm region");
    }
}

kd_selector_t<lrn_gpu::kernel_data, lrn_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> lrn_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_yxfb_lrn_kernel },
    { std::make_tuple(data_types::f16, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_yxfb_lrn_kernel },
    { std::make_tuple(data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_bfyx_lrn_kernel },
    { std::make_tuple(data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_bfyx_lrn_kernel },
};


namespace {
    struct attach {
        attach() {
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), lrn_gpu::create);
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), lrn_gpu::create);
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), lrn_gpu::create);
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), lrn_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
