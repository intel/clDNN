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
#include "kd_selector.h"
#include "implementation_map.h"

#include "api/CPP/data.hpp"

#include <algorithm>

using namespace cldnn;

namespace neural
{
// Kernel names.
static const std::string kernel_name_across_spatial_bfyx = "normalize_gpu_across_spatial_bfyx";
static const std::string kernel_name_within_spatial_bfyx = "normalize_gpu_within_spatial_bfyx";
static const std::string kernel_name_across_spatial_yxfb = "normalize_gpu_across_spatial_yxfb";
static const std::string kernel_name_within_spatial_yxfb = "normalize_gpu_within_spatial_yxfb";

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::configurations>
{
    static constexpr neural::gpu::engine_info_internal::configurations value = neural::gpu::engine_info_internal::configurations::GT_UNKNOWN;
};


struct normalize_gpu : typed_primitive_impl<normalize>
{
    const normalize_node& outer;
    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, normalize_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

	normalize_gpu(const normalize_node& arg)
        : outer(arg),
        _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info()),
        _kernel_data(ks.get_kernel(
            outer,
            outer.input().get_output_layout().data_type,
            outer.input().get_output_layout().format,
            outer.input().get_output_layout().size.batch[0],
            _engine_info.architecture,
            _engine_info.configuration)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_default(const normalize_node& arg)
    {
        auto input_layout = arg.input().get_output_layout();  // input

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;

        // Checking for supported paddings.
        if (input_layout.data_padding.filling_value() != 0.0f)
            throw std::runtime_error("Unknown padding mode in normalize");

        kd.gws0 = input_layout.size.batch[0];
        kd.gws1 = 1;
        kd.gws2 = 1;

        kd.lws0 = kd.gws0;
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    static gpu::jit_constants get_jit_constants(const normalize_node& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_padding = outer.input().get_primitive()->output_padding;
        auto input_size = outer.input().get_output_layout().size;

		auto output_padding = outer.get_primitive()->output_padding;
		auto output_size = outer.get_output_layout().size;

		auto scale_feature_size = outer.scale().get_output_layout().size.spatial[0];

        gpu::jit_constants mem_consts {
            gpu::make_jit_constant("INPUT",                         input_size),
			gpu::make_jit_constant("SCALE_INDEX",					(scale_feature_size == 1) ? "0" : "f"),
            gpu::make_jit_constant("OUTPUT",                        outer.get_output_layout().size),
			gpu::make_jit_constant("EPSILON",						outer.get_primitive()->epsilon),
            gpu::make_jit_constant("FP16_SUPPORTED",                static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",                static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                     data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("INPUT_PADDING",                 input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",                outer.get_primitive()->output_padding),
            gpu::make_jit_constant("INPUT_BUFFER_SIZE_X",           !input_padding ? input_size.spatial[0] : input_size.spatial[0] + input_padding.upper_size().spatial[0] + input_padding.lower_size().spatial[0]),
            gpu::make_jit_constant("INPUT_BUFFER_SIZE_Y",           !input_padding ? input_size.spatial[1] : input_size.spatial[1] + input_padding.upper_size().spatial[1] + input_padding.lower_size().spatial[1]),
			gpu::make_jit_constant("OUTPUT_BUFFER_SIZE_X",          !output_padding ? output_size.spatial[0] : output_size.spatial[0] + output_padding.upper_size().spatial[0] + output_padding.lower_size().spatial[0]),
			gpu::make_jit_constant("OUTPUT_BUFFER_SIZE_Y",          !output_padding ? output_size.spatial[1] : output_size.spatial[1] + output_padding.upper_size().spatial[1] + output_padding.lower_size().spatial[1])
        };

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, normalize_inst& instance) override
    {
        const auto& kd    = _kernel_data;

        return _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
            ({{ kd.gws0, kd.gws1, kd.gws2 }, { kd.lws0, kd.lws1, kd.lws2 }},
                events,
                instance.input_memory(),
                instance.output_memory(),
				instance.scale_memory());
    }


    static primitive_impl* create(const normalize_node& arg) { return new normalize_gpu(arg); }

};


normalize_gpu::kernel_data get_kernel_across_spatial(const normalize_node& arg, format format)
{	
	normalize_gpu::kernel_data kd = normalize_gpu::set_default(arg);

	if (format == cldnn::format::bfyx)
	{
		kd.kernel_name = kernel_name_across_spatial_bfyx;	
	}
	else if (format == cldnn::format::yxfb)
	{
		kd.kernel_name = kernel_name_across_spatial_yxfb;
	}
	else
	{
		throw std::invalid_argument("Unsupported normalize across spatial format - " + cldnn::format::traits(format).order);
	}
 
	return kd;
}

normalize_gpu::kernel_data get_kernel_within_spatial(const normalize_node& arg, format format)
{
	normalize_gpu::kernel_data kd = normalize_gpu::set_default(arg);
   
	if (format == cldnn::format::bfyx)
	{
		kd.kernel_name = kernel_name_within_spatial_bfyx;	
	}
	else if (format == cldnn::format::yxfb)
	{
		kd.kernel_name = kernel_name_within_spatial_yxfb;
	}
	else
	{
		throw std::invalid_argument("Unsupported normalize within spatial format - " + cldnn::format::traits(format).order);
	}
	auto input_layout = arg.input().get_output_layout();
	kd.gws0 = align_to(input_layout.size.spatial[0], 32);
	kd.gws1 = input_layout.size.spatial[1];
	kd.gws2 = input_layout.size.batch[0];

	kd.lws0 = 32;
	kd.lws1 = 1;
	kd.lws2 = 1;

    return kd;
}

normalize_gpu::kernel_data get_bfyx_normalize_kernel(const normalize_node& arg)
{
	if (arg.get_primitive()->across_spatial)
	{
		return get_kernel_across_spatial(arg, cldnn::format::bfyx);
	}
	else
	{
		return get_kernel_within_spatial(arg, cldnn::format::bfyx);
	}
}

normalize_gpu::kernel_data get_yxfb_normalize_kernel(const normalize_node& arg)
{
	if (arg.get_primitive()->across_spatial)
	{
		return get_kernel_across_spatial(arg, cldnn::format::yxfb);
	}
	else
	{
		return get_kernel_within_spatial(arg, cldnn::format::yxfb);
	}
}

kd_selector_t<normalize_gpu::kernel_data, normalize_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> normalize_gpu::ks = 
{
    { std::make_tuple(data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_bfyx_normalize_kernel },
    { std::make_tuple(data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_bfyx_normalize_kernel },
	{ std::make_tuple(data_types::f32, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_yxfb_normalize_kernel },
	{ std::make_tuple(data_types::f16, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), get_yxfb_normalize_kernel },
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
