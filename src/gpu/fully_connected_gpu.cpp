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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "fully_connected_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "network_impl.h"

#include "api/CPP/reorder.hpp"
#include "api/CPP/input_layout.hpp"

using namespace cldnn;

namespace neural
{

struct fully_connected_gpu : typed_primitive_impl<fully_connected>
{
    const fully_connected_node& outer;
    std::vector<network_impl::ptr> _reorders;   // TODO: move this reorder to graph compiler
    gpu::kernel _kernel;

    fully_connected_gpu(const fully_connected_node& arg, const kernel_selector::kernel_data& kd, std::vector<network_impl::ptr> reorders)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
        _reorders = reorders;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, fully_connected_inst& instance) override
    {
        const auto* input_mem = &instance.input_memory();

        if (_reorders.empty() == false)
        {
            auto network = _reorders[0];
            network->set_input_data("input", api_cast(input_mem->get()));
            network->execute(events);
            auto output_id = network->get_output_ids()[0];
            input_mem = &network->get_primitive(output_id)->output_memory();
        }

        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { input_mem };
        args.output = &instance.output_memory();
        args.weights = &instance.weights_memory();
        args.bias = instance.bias_term() ? &instance.bias_memory() : nullptr;;

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const fully_connected_node& arg)
    {
        auto fc_params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(arg);
        auto fc_optional_params = get_default_weights_bias_optional_params<kernel_selector::fully_connected_optional_params>(arg.get_program());
        fc_optional_params.allowReorderInput = true;

        convert_activation_func_params(arg.get_primitive(), fc_params);

        fc_params.output = fc_params.output.FlattenFeatureAndSpatials();

        const auto primitive = arg.get_primitive();

        auto& kernel_selector = kernel_selector::fully_connected_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(fc_params, fc_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        const auto& new_fc_params = *static_cast<kernel_selector::fully_connected_params*>(best_kernels[0].params.get());
        std::vector<network_impl::ptr> reorders; 
        if (fc_params.inputs[0].GetLayout() != new_fc_params.inputs[0].GetLayout())
        {
            const auto& input_layout = arg.input().get_output_layout();
            cldnn::topology topology(
                cldnn::input_layout("input", input_layout),
                cldnn::reorder("reorder", "input", to_data_layout(new_fc_params.inputs[0].GetLayout()), input_layout.data_type)
            );

            reorders.push_back({ arg.get_program().get_engine()->build_network(*api_cast(topology.get()), cldnn::build_options()), false });
        }

        auto fc = new fully_connected_gpu(arg, best_kernels[0], reorders);

        return fc;
    };
};


namespace {
    struct attach {
        attach() {
            auto val_fw = fully_connected_gpu::create;

            implementation_map<fully_connected>::add({
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw },
                { std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
