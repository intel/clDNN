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
#include "implementation_map.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural 
{

struct activation_gpu : typed_primitive_impl<activation>
{
    const activation_node& outer;
    gpu::kernel _kernel;

    activation_gpu(const activation_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, activation_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();
        if (outer.is_parameterized())
        {
            args.slope = &instance.slope_memory();
        }

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const activation_node& arg) 
    { 
        auto activation_params = get_default_params<kernel_selector::activation_params>(arg);
        auto activation_optional_params = get_default_optional_params<kernel_selector::activation_optional_params>(arg.get_program());

        convert_new_activation_func(arg.get_primitive(), activation_params);

        if (arg.is_parameterized())
        {
            const auto& slope_layout = arg.slope_input().get_output_layout();
            const auto& output_layout = arg.get_output_layout();

            const auto params_num = KernelSelector::GetActivationAdditionalParamsNumber(activation_params.activationFunc);

            if (slope_layout.size.count() < static_cast<size_t>(output_layout.size.feature[0] * params_num))
            {
                throw std::runtime_error("Error - not enough data inside additional params buffer");
            }

            activation_params.actParams.inputNlParams.push_back(convert_data_tensor(slope_layout));
        }

        auto& kernel_selector = kernel_selector::activation_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(activation_params, activation_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto activation = new activation_gpu(arg, best_kernels[0]);

        return activation;
    };
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
