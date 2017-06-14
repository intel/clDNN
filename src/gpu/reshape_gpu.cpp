/*
// Copyright (c) 2017 Intel Corporation
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

#include "reshape_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"
#include "events_waiter.h"
#include "network_impl.h"

using namespace cldnn;

namespace neural
{

static const std::string reshape_padding_kernel = "reshape_padding";

struct reshape_gpu : public typed_primitive_impl<reshape>
{
    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        //size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
        bool fp16_supported;
    } _kernel_data;

    std::unique_ptr<gpu::kernel> _kernel;

    static kernel_data get_kernel_data(reshape_node const& node)
    {
        kernel_data data;

        auto input_layout = node.input().get_output_layout();
        auto input_layout_sizes = input_layout.size.sizes(input_layout.format);

        data.gws0 = input_layout_sizes[3] * input_layout_sizes[2];
        data.gws1 = input_layout_sizes[1];
        data.gws2 = input_layout_sizes[0];

        /*data.lws0 = 4;
        data.lws1 = 4;
        data.lws2 = 1;*/

        data.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;
        data.fp16_supported = node.get_program().get_engine()->get_engine_info().supports_fp16 != 0;

        data.kernel_name = reshape_padding_kernel;

        return data;
    }

    static gpu::jit_constants get_jit_constants(reshape_node const& node, kernel_data const& data)
    {
        auto input_layout = node.input().get_output_layout();
        auto output_layout = node.get_output_layout();
        auto mem_format = input_layout.format;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT_SIZES", gpu::get_accumulated_sizes_array(input_layout)),
            gpu::make_jit_constant("OUTPUT_SIZES", gpu::get_accumulated_sizes_array(output_layout)),
            gpu::make_jit_constant("INPUT_PADDING_LOWER", gpu::get_tensor_array(mem_format, input_layout.data_padding.lower_size())),
            gpu::make_jit_constant("OUTPUT_PADDING_LOWER", gpu::get_tensor_array(mem_format, output_layout.data_padding.lower_size())),
            gpu::make_jit_constant("INPUT_BUFFER_SIZES", gpu::get_accumulated_buffer_sizes_array(input_layout)),
            gpu::make_jit_constant("OUTPUT_BUFFER_SIZES", gpu::get_accumulated_buffer_sizes_array(output_layout)),
            gpu::make_jit_constant("UNIT_TYPE", data.fp16_unit_used ? "ushort" : "float")
        };

        return mem_consts;
    }

    static std::unique_ptr<gpu::kernel> create_padding_kernel_or_null(reshape_node const& node, kernel_data& data)
    {
        if (node.is_in_place())
            return nullptr;

        data = get_kernel_data(node);

        //if (!data.fp16_supported && data.fp16_unit_used)
        //    throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        return std::make_unique<gpu::kernel>(node.get_program().get_engine()->get_context(), data.kernel_name, get_jit_constants(node, data));
    }

    reshape_gpu(reshape_node const& node)
        : _kernel(create_padding_kernel_or_null(node, _kernel_data))
    {
    }

    event_impl::ptr execute_impl(std::vector<event_impl::ptr> const& events, reshape_inst& instance)
    {
        if (!_kernel)
        {
            if (events.size() == 1)
                return events[0];

            neural::gpu::events_waiter events_waiter(instance.get_network().get_engine()->get_context());
            return events_waiter.run(events);
        }

        const auto& kd = _kernel_data;

        return _kernel->run<gpu::input_mem, gpu::output_mem>(
        { { kd.gws0, kd.gws1, kd.gws2 } },
            events,
            instance.input_memory(),
            instance.output_memory());
    }

    static primitive_impl* create(reshape_node const& node) { return new reshape_gpu(node); }
};

namespace {
    struct attach {
        attach() {
            implementation_map<reshape>::add({
                { cldnn::engine_types::ocl, reshape_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
