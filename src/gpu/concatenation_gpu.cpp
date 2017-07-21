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

#include "concatenation_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "events_waiter.h"
#include "kernel_selector_helper.h"
#include <initializer_list>

using namespace cldnn;

namespace neural
{

struct concatenation_gpu : typed_primitive_impl<concatenation>
{
    const concatenation_node& outer;
    const concatenation::concatenation_axis concat_axis;

    gpu::engine_info_internal _engine_info;

    std::vector<std::pair<gpu::kernel, kernel_selector::kernel_data>> _kernels;

    concatenation_gpu(const concatenation_node& outer, const std::vector<kernel_selector::kernel_data>& kds)
        : outer(outer)
        , concat_axis(outer.get_primitive()->axis)
        , _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info())
    {
        auto context = outer.get_program().get_engine()->get_context();

        const size_t inputs_count = outer.inputs_count();

        if (!outer.can_be_optimized())
        {
            if (inputs_count != kds.size())
            {
                throw std::runtime_error("Error - not enough kernels for concatenation");
            }

            _kernels.reserve(inputs_count);
            for (size_t i = 0; i < kds.size(); ++i)
            {
                gpu::kernel kernel(outer.get_program().get_engine()->get_context(), kds[i].kernels[0].kernelString);
                _kernels.emplace_back(std::move(kernel), kds[i]);
            }
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, concatenation_inst& instance) override
    {
        if (outer.can_be_optimized())
        {
            if (events.size() == 1)
                return events[0];

            return neural::gpu::events_waiter(outer.get_program().get_engine()->get_context()).run(events);
        }

        assert(outer.inputs_count() == _kernels.size());

        gpu::kernel::kernel_arguments_data args;
        args.output = &instance.output_memory();

        auto tmp_events = events;
        for (size_t i = 0; i < _kernels.size(); ++i)
        {
            args.inputs = { &instance.input_memory(i) };
            args.scalars = &_kernels[i].second.kernels[0].scalars;

            auto event = _kernels[i].first.run(_kernels[i].second.kernels[0], events, args);

            tmp_events.clear();
            tmp_events.push_back(event);
        }
        return tmp_events.at(0);
    }

    static kernel_selector::concat_axis convert_axis(concatenation::concatenation_axis axis)
    {
        switch (axis)
        {
        case concatenation::along_x: return kernel_selector::concat_axis::X;
        case concatenation::along_y: return kernel_selector::concat_axis::Y;
        case concatenation::along_f: return kernel_selector::concat_axis::FEATURE;
        case concatenation::along_b: return kernel_selector::concat_axis::BATCH;
        default: 
            return kernel_selector::concat_axis::X;
        }
    }

    static primitive_impl* create(const concatenation_node& arg) 
    { 
        std::vector<kernel_selector::kernel_data> kds;
        if (!arg.can_be_optimized())
        {
            auto concat_params = get_default_params<kernel_selector::concatenation_params>(arg);
            auto concat_optional_params = get_default_optional_params<kernel_selector::concatenation_optional_params>(arg.get_program());
            auto axis = arg.get_primitive()->axis;
            concat_params.concatParams.axis = convert_axis(axis);

            auto& kernel_selector = kernel_selector::concatenation_kernel_selector::Instance();
            
            int last_offset = 0;

            // TODO: implement one call to kernel selector which provide multiple kernels.
            for (size_t i = 0; i < arg.inputs_count(); ++i)
            {
                const layout& input_layout = arg.input(i).get_output_layout();

                std::vector<tensor::value_type> offest_vec = { 0,0,0,0 };
                offest_vec[axis] = last_offset;
                tensor offset{ std::move(offest_vec) };

                concat_params.inputs[0] = convert_data_tensor(input_layout);
                concat_params.output    = convert_data_tensor(arg.get_output_layout(), 1, offset);

                auto best_kernels = kernel_selector.GetBestKernels(concat_params, concat_optional_params);
                if (best_kernels.empty())
                {
                    throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
                }

                kds.push_back(best_kernels[0]);

                last_offset += input_layout.size.raw[axis];
            }
        }

        auto concat = new concatenation_gpu(arg, kds);

        return concat;
    };
};

namespace {
    struct attach {
        attach() {
            implementation_map<concatenation>::add({
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), concatenation_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), concatenation_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), concatenation_gpu::create },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), concatenation_gpu::create }
            });
        }
        ~attach() {}
    };
}

attach attach_impl;

} // namespace neural
