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

#include "normalize_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "normalize/normalize_kernel_selector.h"
#include "kernel_selector_helper.h"

#include <algorithm>

using namespace cldnn;

namespace neural
{

struct normalize_gpu : typed_primitive_impl<normalize>
{
    const normalize_node& outer;
    gpu::kernel _kernel;

    normalize_gpu(const normalize_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, normalize_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars        = &_kernel_data.kernels[0].scalars;
        args.inputs         = { &instance.input_memory() };
        args.output         = &instance.output_memory();
        args.scale_table    = &instance.scale_memory();

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }


    static primitive_impl* create(const normalize_node& arg) 
    { 
        auto norm_params = get_default_params<kernel_selector::normalize_params>(arg);
        auto norm_optional_params = get_default_optional_params<kernel_selector::normalize_optional_params>(arg.get_program());

        const auto& scale_layout  = arg.scale().get_output_layout();

        norm_params.normParams.normMode = 
            arg.get_primitive()->across_spatial ?
            kernel_selector::normalize_mode::ACROSS_SPATIAL :
            kernel_selector::normalize_mode::WITHIN_SPATIAL;
        norm_params.normParams.epsilon = arg.get_primitive()->epsilon;
        norm_params.normParams.scaleTable = convert_data_tensor(scale_layout).FlattenFeatureAndSpatials();

        auto& kernel_selector = kernel_selector::normalize_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(norm_params, norm_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto lrn = new normalize_gpu(arg, best_kernels[0]);

        return lrn;
    }

};

namespace {
    struct attach 
    {
        attach() 
        {
            implementation_map<normalize>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), normalize_gpu::create);
            implementation_map<normalize>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), normalize_gpu::create);
            implementation_map<normalize>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), normalize_gpu::create);
            implementation_map<normalize>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), normalize_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
