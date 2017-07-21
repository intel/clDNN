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
#include "implementation_map.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct eltwise_gpu : typed_primitive_impl<eltwise>
{
    const eltwise_node& outer;
    gpu::kernel _kernel;

    eltwise_gpu(const eltwise_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    static inline kernel_selector::eltwise_mode convect_to_eltwise_mode(eltwise_mode mode)
    {
        switch (mode)
        {
        case eltwise_mode::sum:  return kernel_selector::eltwise_mode::ADD;
        case eltwise_mode::sub:  return kernel_selector::eltwise_mode::SUB;
        case eltwise_mode::max:  return kernel_selector::eltwise_mode::MAX;
        case eltwise_mode::prod: return kernel_selector::eltwise_mode::MUL;
        default:
            return kernel_selector::eltwise_mode::ADD;
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, eltwise_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory(), &instance.input2_memory() };
        args.output = &instance.output_memory();

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const eltwise_node& arg) 
    { 
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.inputs.push_back(convert_data_tensor(arg.input2().get_output_layout()));
        
        const auto& primitive = arg.get_primitive();
        convert_activation_func_params(primitive, ew_params);

        ew_params.eltwiseParams.operations.push_back({ 
            { kernel_selector::eltwise_params::InputType::Buffer(0), kernel_selector::eltwise_params::InputType::Buffer(1) },
            convect_to_eltwise_mode(primitive->mode) });

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto eltwise = new eltwise_gpu(arg, best_kernels[0]);

        return eltwise;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<eltwise>::add({
                { std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), eltwise_gpu::create },
                { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), eltwise_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
