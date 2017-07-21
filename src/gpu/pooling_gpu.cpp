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

#include "pooling_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

namespace neural
{

struct pooling_gpu : typed_primitive_impl<pooling>
{
    const pooling_node& outer;
    gpu::engine_info_internal _engine_info;
    gpu::kernel _kernel;

    pooling_gpu(const pooling_node &arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _engine_info(arg.get_program().get_engine()->get_context()->get_engine_info())
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, pooling_inst& instance) override
    {
        const auto* input_mem = &instance.input_memory();
        const auto* output_mem = &instance.output_memory();

        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { input_mem };
        args.output = output_mem;

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static void validate(const pooling_node& arg)
    {
        auto const& input_buffer_size = arg.input().get_output_layout().get_buffer_size();
        auto const& input_dimensions = input_buffer_size.batch.size() + input_buffer_size.feature.size() + input_buffer_size.spatial.size();
        auto const& output_buffer_size = arg.get_output_layout().get_buffer_size();
        auto const& output_dimensions = output_buffer_size.batch.size() + output_buffer_size.feature.size() + output_buffer_size.spatial.size();
        auto const& input_format = arg.input().get_output_layout().format;
        auto const& output_format = arg.get_output_layout().format;
        auto& stride = arg.get_primitive()->stride;
        auto const& stride_dimensions = stride.batch.size() + stride.feature.size() + stride.spatial.size();
        auto& window = arg.get_primitive()->size;
        auto const& window_dimensions = window.batch.size() + window.feature.size() + window.spatial.size();

        if (input_dimensions != output_dimensions)
            throw std::invalid_argument("Pooling input/output number of dimension does not match.");

        if (stride_dimensions != output_dimensions)
            throw std::invalid_argument("Pooling stride/output number of dimension does not match.");

        if (window_dimensions != output_dimensions)
            throw std::invalid_argument("Pooling window_size/output number of dimension does not match.");

        if (input_format != output_format)
            throw std::invalid_argument("Pooling input/output data format does not match.");
    }

    static kernel_selector::pool_type cldnn_2_pool_type(cldnn::pooling_mode mode)
    {
        switch (mode)
        {
        case cldnn::pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case cldnn::pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case cldnn::pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
        }
    }

    static kernel_selector::kernel_divider_mode cldnn_2_kernel_divider_mode(cldnn::pooling_mode mode)
    {
        switch (mode)
        {
        case cldnn::pooling_mode::max:
            return kernel_selector::kernel_divider_mode::DONT_CARE;
        case cldnn::pooling_mode::average:
            return kernel_selector::kernel_divider_mode::FIXED;
        case cldnn::pooling_mode::average_no_padding:
            return kernel_selector::kernel_divider_mode::DYNAMIC;
        default:
            assert(0);
            return kernel_selector::kernel_divider_mode::DONT_CARE;
        }
    }

    static primitive_impl* create(const pooling_node& arg)
    {
        validate(arg);

        auto pool_params            = get_default_params<kernel_selector::pooling_params>(arg);
        auto pool_optional_params   = get_default_optional_params<kernel_selector::pooling_optional_params>(arg.get_program());

        const auto primitive        = arg.get_primitive();
        const auto& stride          = primitive->stride;
        const auto& input_offset    = primitive->input_offset;

        auto& pp                    = pool_params.poolParams;

        pp.poolType                 = cldnn_2_pool_type(primitive->mode);
        pp.remainderAction          = kernel_selector::pool_remainder::CEIL;
        pp.divMode                  = cldnn_2_kernel_divider_mode(primitive->mode);
        
        pp.poolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
        };

        pp.poolPad = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        pp.poolStride = {
            (uint32_t)stride.spatial[0],
            (uint32_t)stride.spatial[1]
        };

        auto& kernel_selector   = kernel_selector::pooling_kernel_selector::Instance();
        auto best_kernels       = kernel_selector.GetBestKernels(pool_params, pool_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto pool = new pooling_gpu(arg, best_kernels[0]);

        return pool;
    }
};

namespace
{

    struct attach
    {
        attach()
        {
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), pooling_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
