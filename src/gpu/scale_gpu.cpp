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

#include "scale_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include <algorithm>

using namespace cldnn;

namespace neural
{
// Kernel names.
static const std::string kernel_name = "scale_gpu";

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

struct scale_gpu : typed_primitive_impl<scale>
{
    const scale_node& outer;

    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
        bool fp16_supported;
        bool input_bfyx_used; ///< Indicates that bfyx format of input is used.
        bool scale_bfyx_used; ///< Indicates that bfyx format of scale is used.
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, scale_node, data_types, format::type, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    scale_gpu(const scale_node& arg) :
        outer(arg),
        _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(
            outer,
            outer.input().get_output_layout().data_type,
            outer.input().get_output_layout().format,
            _engine_info.architecture,
            _engine_info.configuration)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data))
    {}

    static kernel_data set_kernel_data(const scale_node& outer)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        auto input_layout = outer.input().get_output_layout();
        auto const& input_size = input_layout.get_buffer_size();

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;
        kd.fp16_supported = engine_info.supports_fp16 != 0;
        kd.scale_bfyx_used = false;
        kd.input_bfyx_used = false;
        // Determine global work sizes.
        kd.gws0 = input_size.batch[0];   // B
        kd.gws1 = input_size.feature[0]; // F
        kd.gws2 = input_size.spatial[0] * input_size.spatial[1]; // X, Y
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

    static gpu::jit_constants get_jit_constants(const scale_node& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_size = outer.input().get_output_layout().size;
        auto output_size = outer.get_output_layout().size;
        auto scale_size = outer.scale_in().get_output_layout().size;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                 input_size),
            gpu::make_jit_constant("OUTPUT",                output_size),
            gpu::make_jit_constant("SCALE",                 scale_size),
            gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("BIAS_TERM",             static_cast<int>(!outer.get_primitive()->bias.empty())),
            gpu::make_jit_constant("SCALE_BFYX_USED",       static_cast<int>(data.scale_bfyx_used)),
            gpu::make_jit_constant("INPUT_BFYX_USED",       static_cast<int>(data.input_bfyx_used)),
            gpu::make_jit_constant("INPUT_PADDING",         outer.input().get_output_layout().data_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",        outer.get_output_layout().data_padding)
        };

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, scale_inst& instance) override
    {
        const auto& kd = _kernel_data;

        const auto& input_mem = instance.input_memory();  // input
        const auto& scale_mem = instance.scale_memory();  // scale_input
        const auto& output_mem = instance.output_memory(); // output

        if (instance.bias_term())
        {
            const auto& bias_mem = instance.bias_memory();  // bias
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem>({ { kd.gws0, kd.gws1, kd.gws2 },{ kd.lws0, kd.lws1, kd.lws2 } }, events, input_mem, output_mem, scale_mem, bias_mem);
        }
        else
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>({ {kd.gws0, kd.gws1, kd.gws2 }, {kd.lws0, kd.lws1, kd.lws2 } }, events, input_mem, output_mem, scale_mem);
    }

    static primitive_impl* create(const scale_node& arg) { return new scale_gpu(arg); };
};

scale_gpu::kernel_data set_default(const scale_node& arg)
{
    scale_gpu::kernel_data kd = scale_gpu::set_kernel_data(arg);
    kd.scale_bfyx_used = (arg.scale_in().get_output_layout().format == cldnn::format::bfyx) ? true : false;
    kd.input_bfyx_used = (arg.input().get_output_layout().format == cldnn::format::bfyx) ? true : false;
    kd.kernel_name = kernel_name;

    return kd;
}

kd_selector_t<scale_gpu::kernel_data, scale_node, data_types, format::type, kd_optional_selector_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> scale_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::yxfb, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f32, format::bfyx, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::bfyx, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
};

namespace {
    struct attach {
        attach() {
            auto val_fw = scale_gpu::create;

            implementation_map<scale>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<scale>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<scale>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<scale>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} // namespace neural
