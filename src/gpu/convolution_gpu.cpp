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

#include "convolution_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"

namespace cldnn { namespace gpu {

struct convolution_gpu : typed_primitive_gpu_impl<convolution>
{
    using parent = typed_primitive_gpu_impl<convolution>;
    using parent::parent;

protected:

    virtual bool validate(typed_primitive_inst<convolution>& instance) const override
    {
        bool res = parent::validate(instance);

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory().get_layout().data_type, "output memory", instance.output_memory().get_layout().data_type, "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_outer.id(), "Input memory", instance.input_memory().get_layout().data_type, "filter memory", instance.weights_memory(0).get_layout().data_type, "");

        return res;
    }

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<convolution>& instance, int32_t split) const override
    {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights    = &instance.weights_memory(split);
        args.bias       = instance.bias_term() ? &instance.bias_memory(split) : nullptr;

        return args;
    }

    virtual int32_t get_split() const override
    { 
        return _outer.get_split(); 
    }

public:

    static primitive_impl* create(const convolution_node &arg)
    {
        const auto& primitive       = arg.get_primitive();
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& weights_layout  = arg.weights(0).get_output_layout();

        const auto& input_size      = input_layout.size;
        const auto& weights_size    = weights_layout.size;

        const auto& split           = primitive->split();
        const auto& stride          = primitive->stride;
        const auto& dilation        = primitive->dilation;
        const auto& input_offset    = primitive->input_offset;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        assert(arg.get_output_layout().size.feature[0] / primitive->split() == weights_layout.size.batch[0]);

        auto conv_params = get_weights_bias_default_params<kernel_selector::convolution_params>(arg, actual_split);
        auto conv_optional_params = get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        const auto additional_offset = tensor::max(input_offset, 0);
        if (additional_offset != 0)
        {
            conv_params.inputs[0] = convert_data_tensor(input_layout, actual_split, additional_offset);
        }

        if(primitive->with_activation)
            convert_activation_func_params(primitive, conv_params);

        if (input_layout.format == format::winograd_2x3_s1_data)
        {
            conv_params.convParams.winograd_tile_n = 4;
            conv_params.convParams.winograd_tile_m = 8;
            conv_params.convParams.winograd_input_tile_width = 4;
            conv_params.convParams.winograd_input_tile_height = 1;
        }

        conv_params.convParams.depthwiseSeparableOpt = depthwise_separable_opt;

        conv_params.convParams.split = split;
        conv_params.convParams.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
        };

        conv_params.convParams.padding = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        conv_params.convParams.stride = {
            (uint32_t)std::min(stride.spatial[0], input_size.spatial[0]),
            (uint32_t)std::min(stride.spatial[1], input_size.spatial[1])
        };
        conv_params.convParams.dilation = {
            (uint32_t)std::min(dilation.spatial[0], input_size.spatial[0]),
            (uint32_t)std::min(dilation.spatial[1], input_size.spatial[1])
        };

        auto& kernel_selector = kernel_selector::convolution_kernel_selector::Instance();

        const auto& tuning_config = arg.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache)
        {
            conv_optional_params.tuningParams.runner = std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), true);
        }

        KernelSelector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);
		
        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto conv = new convolution_gpu(arg, best_kernels[0]);

        return conv;
    }
};

namespace{
    struct attach {
        attach() {
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::winograd_2x3_s1_data), convolution_gpu::create);
            implementation_map<convolution>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::winograd_2x3_s1_data), convolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
