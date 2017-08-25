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
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "error_handler.h"

namespace cldnn { namespace gpu {

struct reorder_gpu : typed_primitive_gpu_impl<reorder>
{
    using parent = typed_primitive_gpu_impl<reorder>;
    using parent::parent;

protected:

    virtual bool optimized_out(reorder_inst& instance) const override
    {
        return
            parent::optimized_out(instance) || _outer.can_be_optimized();
    }

    virtual kernel::kernel_arguments_data get_arguments(reorder_inst& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        if (_outer.has_mean())
        {
            args.bias = &instance.mean_memory();
        }

        return args;
    }

public:

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

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto reorder = new reorder_gpu(arg, best_kernels[0]);

        return reorder;
    }
};

namespace {
    struct attach {
        attach() {
            implementation_map<reorder>::add({
                { engine_types::ocl, reorder_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }