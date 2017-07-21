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

#include "convolution_inst.h"
#include "kernel.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include <initializer_list>

using namespace cldnn;

namespace neural 
{

struct convolution_gpu : typed_primitive_impl<convolution> {
    const convolution_node& outer;
    gpu::engine_info_internal _engine_info;
    gpu::kernel _kernel;

    convolution_gpu(const convolution_node &arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _engine_info(arg.get_program().get_engine()->get_context()->get_engine_info())
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events, convolution_inst& instance) override
    {
        auto split = outer.get_primitive()->split();

        const auto* input_mem = &instance.input_memory();
        const auto* output_mem = &instance.output_memory();
        const auto* filter_mem_0 = &instance.weights_memory(0);

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        if (input_mem->get_layout().data_type != output_mem->get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of output.");
        if (input_mem->get_layout().data_type != filter_mem_0->get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of filter.");

        std::vector<event_impl::ptr> tmp_events(events);

        // execute kernels
        for (decltype(split) i = 0; i < split; i++)
        {
            const auto* filter_mem = &instance.weights_memory(i);
            const auto* bias_mem = instance.bias_term() ? &instance.bias_memory(i) : nullptr;

            gpu::kernel::kernel_arguments_data args;
            args.scalars = &_kernel_data.kernels[0].scalars;
            args.inputs = { input_mem };
            args.output = output_mem;
            args.weights = filter_mem;
            args.bias = bias_mem;
            args.split = i;

            auto event = _kernel.run(_kernel_data.kernels[0], tmp_events, args);

            tmp_events.clear();
            tmp_events.emplace_back(event);
        }

        return tmp_events.at(0);
    }

    static primitive_impl* create(const convolution_node &arg)
    {
        const auto& primitive       = arg.get_primitive();
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& weights_layout  = arg.weights(0).get_output_layout();

        const auto& input_size      = input_layout.size;
        const auto& weights_size    = weights_layout.size;

        const auto& split           = primitive->split();
        const auto& stride          = primitive->stride;
        const auto& dilation        = primitive->dilation;
        const auto& input_offset    = primitive->input_offset;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto conv_params = get_weights_bias_default_params<kernel_selector::convolution_params>(arg, split);
        auto conv_optional_params = get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        convert_activation_func_params(primitive, conv_params);

        conv_params.convParams.split = split;
        conv_params.convParams.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        conv_params.convParams.padding = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        conv_params.convParams.stride = {
            (uint32_t)std::min(stride.spatial[0], input_size.spatial[0]),
            (uint32_t)std::min(stride.spatial[1], input_size.spatial[1])
        };
        conv_params.convParams.dilation = {
            (uint32_t)std::min(dilation.spatial[0], input_size.spatial[0]),
            (uint32_t)std::min(dilation.spatial[1], input_size.spatial[1])
        };

        auto& kernel_selector = kernel_selector::convolution_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto conv = new convolution_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), convolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
