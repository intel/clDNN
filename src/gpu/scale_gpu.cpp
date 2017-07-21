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
#include "implementation_map.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct scale_gpu : typed_primitive_impl<scale>
{
    const scale_node& outer;
    gpu::kernel _kernel;

    scale_gpu(const scale_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd; 
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, scale_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory(), &instance.scale_memory() };
        args.output = &instance.output_memory();

        if (outer.bias_term())
        {
            args.inputs.push_back(&instance.bias_memory());
        }

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const scale_node& arg) 
    { 
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.inputs.push_back(convert_data_tensor(arg.scale_in().get_output_layout()));

        ew_params.eltwiseParams.operations.push_back({
            { kernel_selector::eltwise_params::InputType::Buffer(0), kernel_selector::eltwise_params::InputType::Buffer(1) },
            kernel_selector::eltwise_mode::MUL });

        if (arg.bias_term())
        {
            ew_params.inputs.push_back(convert_data_tensor(arg.bias().get_output_layout()));
            ew_params.eltwiseParams.operations.push_back({
                { kernel_selector::eltwise_params::InputType::Intermediate(0), kernel_selector::eltwise_params::InputType::Buffer(2) },
                kernel_selector::eltwise_mode::ADD });
        }

        ew_params.eltwiseParams.layoutBased = true;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto scale = new scale_gpu(arg, best_kernels[0]);

        return scale;
    }
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
