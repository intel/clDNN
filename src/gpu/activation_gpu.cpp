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

#include "activation_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include <algorithm>

using namespace cldnn;

namespace neural 
{
// Kernel names.
static const std::string kernel_name = "relu_gpu";
static const std::string kernel_name_bfyx = "relu_gpu_bfyx";

struct activation_gpu : typed_primitive_impl<activation>
{
    const activation_node& outer;

    struct kernel_data
    {
        size_t gws0;
        size_t lws0;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;

    gpu::kernel _kernel;

    activation_gpu(const activation_node& arg)
        : outer(arg),
        _kernel_data(set_kernel_data(outer)),
        _kernel(arg.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_kernel_data(const activation_node& outer)
    {
        auto input_layout  = outer.input().get_output_layout();  // input

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;

        // Determine global work sizes.
        kd.gws0 = input_layout.count();
        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        if (input_layout.format == cldnn::format::bfyx)
        {
            kd.kernel_name = kernel_name_bfyx;
        }
        else
        {
            kd.kernel_name = kernel_name;
        }

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const activation_node& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_size = outer.input().get_output_layout().size;
        auto input_padding = outer.input().get_output_layout().data_padding;
        auto output_size = outer.get_output_layout().size;
        auto output_padding = outer.get_output_layout().data_padding;

        bool is_parameterized = outer.is_parameterized();

        gpu::jit_constants mem_consts
        {
            gpu::make_jit_constant("INPUT",                         input_size),
            gpu::make_jit_constant("INPUT_PADDING",                 input_padding),
            gpu::make_jit_constant("OUTPUT",                        output_size),
            gpu::make_jit_constant("OUTPUT_PADDING",                output_padding),
            gpu::make_jit_constant("RELU",                          is_parameterized ? 0 : 1),
            gpu::make_jit_constant("PRELU",                         is_parameterized ? 1 : 0),
            gpu::make_jit_constant("NEGATIVE_SLOPE",                outer.get_primitive()->negative_slope),
            gpu::make_jit_constant("FP16_SUPPORTED",                static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",                static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                     data.fp16_unit_used ? "half" : "float")
        };

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, activation_inst& instance) override
    {
        const auto& kd    = _kernel_data;

        if (outer.is_parameterized())
        {
            return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>(
                { kd.gws0, kd.lws0 },
                events,
                instance.input_memory(),
                instance.output_memory(),
                instance.slope_memory());
        }
        else
        {
            return _kernel.run<gpu::input_mem, gpu::output_mem>(
                { kd.gws0, kd.lws0 },
                events,
                instance.input_memory(),
                instance.output_memory());
        }  
    }

    static primitive_impl* create(const activation_node& arg) { return new activation_gpu(arg); };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = activation_gpu::create;
    
            implementation_map<activation>::add({
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw},
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw},
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
