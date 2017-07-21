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

#include "crop_inst.h"
#include "kernel.h"
#include "events_waiter.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct crop_gpu : typed_primitive_impl<crop>
{
    const crop_node& outer;
    gpu::kernel _kernel;

    crop_gpu(const crop_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, crop_inst& instance) override
    {
        if (outer.can_be_optimized())
        {
            if (events.size() == 1)
                return events[0];

            return neural::gpu::events_waiter(outer.get_program().get_engine()->get_context()).run(events);
        }

        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const crop_node& arg) 
    { 
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg, 1);
        auto ew_optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.eltwiseParams.operations.push_back({{ kernel_selector::eltwise_params::InputType::Buffer(0) }, kernel_selector::eltwise_mode::ASSIGN });

        const auto& input_layout = arg.input().get_output_layout();
        ew_params.inputs[0] = convert_data_tensor(input_layout, 1, arg.get_primitive()->offsets);

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto crop = new crop_gpu(arg, best_kernels[0]);

        return crop;
    };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = crop_gpu::create;

            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<crop>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };

    attach attach_impl;

}
} // namespace neural
