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

#include "eltwise_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

using namespace cldnn;

using namespace cldnn;

namespace neural
{
// Kernel names.
static const std::string kernel_name = "eltwise_gpu";
static const std::string kernel_name_bfyx = "eltwise_gpu_bfyx";

struct eltwise_gpu : typed_primitive_impl<eltwise>
{
    const eltwise_node& outer;

    struct kernel_data
    {
        size_t gws0;
        size_t lws0;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    eltwise_gpu(const eltwise_node& arg)
        : outer(arg),
        _kernel_data(set_kernel_data(outer)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_kernel_data(const eltwise_node& outer)
    {
        auto output_layout = outer.get_output_layout(); // output

        if (outer.has_padded_dependency())
        {
            throw std::runtime_error("Input padding for eltwise not yet supported");
        }

        kernel_data kd;

        kd.fp16_unit_used = output_layout.data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = output_layout.count();
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        if (output_layout.format == cldnn::format::bfyx)
        {
            kd.kernel_name = kernel_name_bfyx;
        }
        else
        {
            if (outer.is_padded())
                throw std::runtime_error("Output padding for eltwise is not yet supported for not-bfyx input");

            kd.kernel_name = kernel_name;
        }

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const eltwise_node& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_layout = outer.input().get_output_layout();
        auto input2_layout = outer.input2().get_output_layout();
        auto output_layout = outer.get_output_layout();

        return{
            gpu::make_jit_constant("INPUT",                 input_layout.size),
            gpu::make_jit_constant("OUTPUT",                output_layout.size),
            gpu::make_jit_constant("INPUT2" ,               input2_layout.size),
            gpu::make_jit_constant("OUTPUT_PADDING",        outer.get_output_layout().data_padding),
            gpu::make_jit_constant("FP16_SUPPORTED",        static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",        static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",             data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("SUM_MODE_USED",         outer.get_primitive()->mode == cldnn::eltwise_mode::sum ? 1 : 0),
            gpu::make_jit_constant("MAX_MODE_USED",         outer.get_primitive()->mode == cldnn::eltwise_mode::max ? 1 : 0),
            gpu::make_jit_constant("SUB_MODE_USED",         outer.get_primitive()->mode == cldnn::eltwise_mode::sub ? 1 : 0),
            gpu::make_jit_constant("PROD_MODE_USED",        outer.get_primitive()->mode == cldnn::eltwise_mode::prod ? 1 : 0),
            gpu::make_jit_constant("RELU",                  static_cast<int>(outer.get_primitive()->with_activation)),
            gpu::make_jit_constant("NEGATIVE_SLOPE",        outer.get_primitive()->activation_negative_slope),
        };
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, eltwise_inst& instance) override
    {
        const auto& kd    = _kernel_data;

        // input2_mem memory in bfyx or yxfb.
        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>(
            {kd.gws0, kd.lws0},
            events,
            instance.input_memory(),
            instance.output_memory(),
            instance.input2_memory());
    }

    static primitive_impl* create(const eltwise_node& outer) { return new eltwise_gpu(outer); }
};

namespace {
    struct attach {
        attach() {
            implementation_map<eltwise>::add({
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), eltwise_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
