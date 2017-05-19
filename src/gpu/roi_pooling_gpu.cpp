/*
// Copyright (c) 2017 Intel Corporation
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

#include "roi_pooling_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include <algorithm>

using namespace cldnn;

namespace neural
{
static const std::string roi_pooling_kernel_name            = "roi_pooling_gpu";

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

struct roi_pooling_gpu : typed_primitive_impl<roi_pooling>
{
    const roi_pooling_node& outer;

    gpu::engine_info_internal _engine_info;

    struct kernel_data 
    {
        size_t gws0, gws1, gws2; ///< Local work sizes (3D).
        size_t lws0, lws1, lws2; ///< Global work sizes (3D).
        std::string kernel_name;
        bool fp16_unit_used;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, roi_pooling_node, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    roi_pooling_gpu(const roi_pooling_node& arg)
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

    static kernel_data set_default(const roi_pooling_node& outer)
    {
        kernel_data kd;

        cldnn::data_types input_dt = outer.input().get_output_layout().data_type;
       
        kd.fp16_unit_used = (input_dt == cldnn::data_types::f16);

        // Determine global work sizes.
        kd.gws0 = outer.get_output_layout().get_buffer_size().count();
        kd.gws1 = 1;
        kd.gws2 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        kd.kernel_name = roi_pooling_kernel_name;

        return kd;
    }


    static gpu::jit_constants get_jit_constants(const roi_pooling_node& outer, const kernel_data& data)
    {
        neural::gpu::engine_info_internal engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        if (!engine_info.supports_fp16 && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        const cldnn::tensor& input_size = outer.input().get_output_layout().get_buffer_size();

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",             input_size),
            gpu::make_jit_constant("POOLED_HEIGHT",     outer.get_primitive()->pooled_height),
            gpu::make_jit_constant("POOLED_WIDTH",      outer.get_primitive()->pooled_width),
            gpu::make_jit_constant("INPUT_PADDING",     outer.input().get_output_layout().data_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",    outer.get_output_layout().data_padding),
            gpu::make_jit_constant("SPATIAL_SCALE",     outer.get_primitive()->spatial_scale),
            gpu::make_jit_constant("FP16_SUPPORTED",    static_cast<int>(engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",    static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",         data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_INIT_VAL_MAX", data.fp16_unit_used ? "HALF_MAX" : "FLT_MAX")
        };
        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, roi_pooling_inst& instance) override
    {
        const kernel_data& kd = _kernel_data;

        const cldnn::memory& input_data = instance.input_memory(); 
        const cldnn::memory& input_rois = instance.rois_memory(); 
        const cldnn::memory& output_mem = instance.output_memory(); 

        return _kernel.run<gpu::input_mem, gpu::input_mem, gpu::output_mem >
          ({{kd.gws0, kd.gws1, kd.gws2}, {kd.lws0, kd.lws1, kd.lws2}}, events, input_data, input_rois, output_mem);
    }

    static primitive_impl* create(const roi_pooling_node& arg)
    {
        auto input_arg  = arg.input().get_output_layout();
        auto output_arg = arg.get_output_layout();

        const auto padding_filling_value = arg.get_output_layout().data_padding.filling_value();

        if (padding_filling_value != 0.0f) {
            throw std::logic_error("ROI pooling supports only zero padding.");
        }

        if (input_arg.format != output_arg.format) {
            throw std::invalid_argument("ROI pooling input/output data format does not match.");
        }
        
        return new roi_pooling_gpu(arg);
    }
};


kd_selector_t<roi_pooling_gpu::kernel_data, roi_pooling_node, 
            data_types,
            format::type,
            kd_optional_selector_t, 
            int, 
            neural::gpu::engine_info_internal::architectures, 
            neural::gpu::engine_info_internal::configurations> roi_pooling_gpu::ks = {
    { std::make_tuple(data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
    { std::make_tuple(data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
};

namespace
{

    struct attach
    {
        attach()
        {
            implementation_map<roi_pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), roi_pooling_gpu::create);
            implementation_map<roi_pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), roi_pooling_gpu::create);
        }

        ~attach()
        {
        }
    };

    attach attach_impl;
}

}