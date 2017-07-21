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

#include "reorder_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct reorder_gpu : typed_primitive_impl<reorder>
{
    const reorder_node& outer;
    gpu::kernel _kernel;

    reorder_gpu(const reorder_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, reorder_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();
        if (outer.has_mean())
        {
            args.bias = &instance.mean_memory();
        }

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const reorder_node& arg)
    {
        auto reorder_params = get_default_params<kernel_selector::reorder_params>(arg);
        auto reorder_optional_params = get_default_optional_params<kernel_selector::reorder_optional_params>(arg.get_program());

        if (arg.has_mean())
        {
            const auto& mean_layout = arg.mean().get_output_layout();
            reorder_params.reorderParams.mean = convert_data_tensor(mean_layout);
            reorder_params.reorderParams.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
        }
        else if (arg.get_primitive()->subtract_per_feature.empty() == false)
        {
            reorder_params.reorderParams.mode = kernel_selector::mean_subtruct_mode::INSIDE_PARAMS;
            reorder_params.reorderParams.meanValues = arg.get_primitive()->subtract_per_feature;
        }
        else
        {
            reorder_params.reorderParams.mode = kernel_selector::mean_subtruct_mode::NONE;
        }

        auto& kernel_selector = kernel_selector::reorder_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reorder_params, reorder_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto reorder = new reorder_gpu(arg, best_kernels[0]);

        return reorder;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<reorder>::add({
                { cldnn::engine_types::ocl, reorder_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}