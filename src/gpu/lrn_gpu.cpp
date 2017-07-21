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

#include "lrn_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{
    
struct lrn_gpu : typed_primitive_impl<lrn>
{
    const lrn_node& outer;
    gpu::engine_info_internal _engine_info;
    gpu::kernel _kernel;

    lrn_gpu(const lrn_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _engine_info(arg.get_program().get_engine()->get_context()->get_engine_info())
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, lrn_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory() };
        args.output = &instance.output_memory();

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const lrn_node& arg) 
    {
        auto lrn_params = get_default_params<kernel_selector::lrn_params>(arg);
        auto lrn_optional_params = get_default_optional_params<kernel_selector::lrn_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();

        lrn_params.lrnParams.alpha      = primitive->alpha;
        lrn_params.lrnParams.beta       = primitive->beta;
        lrn_params.lrnParams.k          = primitive->k;
        lrn_params.lrnParams.localSize  = primitive->size;
        lrn_params.lrnParams.divMode    = kernel_selector::kernel_divider_mode::DONT_CARE;
        lrn_params.lrnParams.normMode   = 
            primitive->norm_region == cldnn_lrn_norm_region_within_channel ? 
            kernel_selector::lrn_mode::WITHIN_CHANNEL :
            kernel_selector::lrn_mode::ACROSS_CHANNEL;

        auto& kernel_selector = kernel_selector::lrn_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(lrn_params, lrn_optional_params);
        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto lrn = new lrn_gpu(arg, best_kernels[0]);

        return lrn;
    }

};

namespace {
    struct attach {
        attach() {
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), lrn_gpu::create);
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), lrn_gpu::create);
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), lrn_gpu::create);
            implementation_map<lrn>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), lrn_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
