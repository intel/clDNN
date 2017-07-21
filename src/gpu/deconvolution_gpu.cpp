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

#include "deconvolution_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"
#include "deconvolution/deconvolution_kernel_selector.h"
#include "kernel_selector_helper.h"
#include <initializer_list>

using namespace cldnn;

namespace neural 
{

struct deconvolution_gpu : typed_primitive_impl<deconvolution> 
{
    const deconvolution_node& outer;
    gpu::kernel _kernel;

    deconvolution_gpu(const deconvolution_node &arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }


    event_impl::ptr execute_impl(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events, deconvolution_inst& instance) override
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

    static primitive_impl* create(const deconvolution_node& arg)
    {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout();

        assert(arg.get_output_layout().size.feature[0] / arg.get_primitive()->split() == weights_layout.size.batch[0]); // memory::format oixy

        switch (weights_layout.fused_format())
        {
            // FP32 (float)
        case fuse(data_types::f32, format::bfyx):
        case fuse(data_types::f32, format::yxfb):
        case fuse(data_types::f16, format::bfyx):
        case fuse(data_types::f16, format::yxfb):
            break;
        default:
            throw std::runtime_error("deconvolution weights format unsupported");
        }

        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& stride = primitive->stride;
#if 0 // TODO: support dilation
        const auto& dilation = primitive->dilation;
#else
        const tensor dilation = {0,0,1,1};
#endif
        const auto& input_offset = primitive->input_offset;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto deconv_params = get_weights_bias_default_params<kernel_selector::deconvolution_params>(arg, split);
        auto deconv_optional_params = get_default_weights_bias_optional_params<kernel_selector::deconvolution_optional_params>(arg.get_program());

        convert_activation_func_params(primitive, deconv_params);

        deconv_params.deconvParams.split = split;
        deconv_params.deconvParams.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        deconv_params.deconvParams.padding = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        deconv_params.deconvParams.stride = {
            (uint32_t)stride.spatial[0],
            (uint32_t)stride.spatial[1]
        };

        deconv_params.deconvParams.dilation = {
            (uint32_t)dilation.spatial[0],
            (uint32_t)dilation.spatial[1]
        };

        auto& kernel_selector = kernel_selector::deconvolution_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(deconv_params, deconv_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto deconv = new deconvolution_gpu(arg, best_kernels[0]);

        return deconv;
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), deconvolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
