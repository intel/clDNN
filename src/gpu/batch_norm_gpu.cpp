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

#include "batch_norm_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "network_impl.h"
#include "implementation_map.h"

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace cldnn;

namespace neural
{

// Kernel names.
static const std::string kernel_name = "batch_norm_gpu";
static const std::string kernel_name_global_stats = "batch_norm_use_global_stats_gpu";

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

struct batch_norm_gpu : typed_primitive_impl<batch_norm>
{
    const batch_norm_node& outer;
    const layout input_layout;
    const layout output_layout;

    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
        bool fp16_supported;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, batch_norm_node, data_types, format::type, bool, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    batch_norm_gpu(const batch_norm_node& arg):
        outer(arg),
        input_layout(arg.input().get_output_layout()),
        output_layout(arg.get_output_layout()),
        _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(
            outer, 
            input_layout.data_type,
            input_layout.format,
            outer.get_primitive()->use_global_stats,
            _engine_info.architecture,
            _engine_info.configuration)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_kernel_data(const batch_norm_node& outer)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        const auto& input_layout = outer.input().get_output_layout();  // input

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;
        kd.fp16_supported = engine_info.supports_fp16 != 0;

        // Determine global work sizes.
        kd.gws0 = input_layout.size.batch[0];   // B
        kd.gws1 = input_layout.size.feature[0]; // F
        kd.gws2 = input_layout.size.spatial[0] * input_layout.size.spatial[1]; // X, Y
        // Find largest positive local work size that is divider for global work size.
        kd.lws2 = std::min(std::max(kd.gws2, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws2 % kd.lws2 != 0)
        {
            --kd.lws2;
        }
        kd.lws1 = 1;
        kd.lws0 = 1;

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const batch_norm_node& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_padding = outer.input().get_output_layout().data_padding;

        gpu::jit_constants mem_consts {
            gpu::make_jit_constant("INPUT",                 outer.input().get_output_layout().size),
            gpu::make_jit_constant("OUTPUT",                outer.get_output_layout().size),
            gpu::make_jit_constant("EPSILON",               data.fp16_unit_used ? 0.0f : outer.get_primitive()->epsilon),
            gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",         data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("UNIT_VAL_SQUARE",       data.fp16_unit_used ? "2.0h" : "2.0f"),
            gpu::make_jit_constant("INPUT_PADDING",         input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",        outer.get_output_layout().data_padding),
            gpu::make_jit_constant("BFYX_USED",             static_cast<int>(outer.input().get_output_layout().format == cldnn::format::bfyx ? true : false)),
        };

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, batch_norm_inst& instance) override
    {
        const auto& kd = _kernel_data;

        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>(
            {{ kd.gws0, kd.gws1, kd.gws2 }, {kd.lws0, kd.lws1, kd.lws2 }},
            events,
            instance.input_memory(),
            instance.output_memory(),
            instance.mean_memory(),
            instance.variance_memory());
    }

    static primitive_impl* create(const batch_norm_node &arg) { return new batch_norm_gpu(arg); };
};

batch_norm_gpu::kernel_data set_default_use_global_stats(const batch_norm_node& arg)
{
    batch_norm_gpu::kernel_data kd = batch_norm_gpu::set_kernel_data(arg);
    kd.kernel_name = kernel_name_global_stats;

    return kd;
}

batch_norm_gpu::kernel_data set_default(const batch_norm_node& arg)
{
    batch_norm_gpu::kernel_data kd = batch_norm_gpu::set_kernel_data(arg);
    kd.kernel_name = kernel_name;

    return kd;
}

kd_selector_t<batch_norm_gpu::kernel_data, batch_norm_node, data_types, format::type, bool, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> batch_norm_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, false, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f32, format::yxfb, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_use_global_stats },
    { std::make_tuple(data_types::f16, format::yxfb, false, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::yxfb, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_use_global_stats },
    { std::make_tuple(data_types::f32, format::bfyx, false, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f32, format::bfyx, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_use_global_stats },
    { std::make_tuple(data_types::f16, format::bfyx, false, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::bfyx, true, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_use_global_stats },
};

namespace {
    struct attach {
        attach() {
            auto val_fw = batch_norm_gpu::create;

            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<batch_norm>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} // namespace neural
