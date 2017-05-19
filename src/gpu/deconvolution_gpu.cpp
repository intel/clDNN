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

#include "deconvolution_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"

#include <initializer_list>

using namespace cldnn;

namespace neural 
{

static const std::string kernel_name_yxfb_oiyx = "deconvolution_gpu_yxfb_oiyx";
static const std::string kernel_name_yxfb_yxio = "deconvolution_gpu_yxfb_yxio";
static const std::string kernel_name_bfyx_oiyx = "deconvolution_gpu_bfyx_oiyx";
static const std::string kernel_name_bfyx_yxio = "deconvolution_gpu_bfyx_yxio";

template <>
struct kd_default_value_selector<neural::gpu::engine_info_internal::architectures>
{
    static constexpr neural::gpu::engine_info_internal::architectures value = neural::gpu::engine_info_internal::architectures::GEN_UNKNOWN;
};

template <>
struct kd_default_value_selector<gpu::engine_info_internal::configurations>
{
    static constexpr gpu::engine_info_internal::configurations value = gpu::engine_info_internal::configurations::GT_UNKNOWN;
};

struct deconvolution_gpu : typed_primitive_impl<deconvolution> {
    const deconvolution_node& outer;
    const layout input_layout;
    const layout weights_layout;
    const layout output_layout;

    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        size_t lws0, lws1, lws2;
        size_t ofm_per_work_item; // how many output feature maps a single work item compute
        size_t batches_per_work_item; // how many batches will a single work item compute
        size_t block_width, block_height; // used for kernels processing blocks
        size_t prefetch;
        size_t input_block_array_size; ///< Number of elements in array of UNIT_TYPE that must be specified in kernel to store/cache input block.
        size_t input_block_width;      ///< Number of elements in X dimension stored/cached in input block.
        std::string kernel_name;       ///< Name of a kernel/algorithm to execute.
        bool fp16_unit_used;           ///< Value indicating that FP16 half precision floating point type will be used (instead of single precision).
        size_t leftovers;
    } _kernel_data;

    gpu::kernel _kernel;

    static kernel_data set_default(const deconvolution_node& arg)
    {
        const auto& input_layout = arg.input().get_output_layout();  // input
        const auto& output_size = arg.get_output_layout().get_buffer_size(); // output

        auto split = arg.get_primitive()->split();
        auto batch_size = output_size.batch[0];

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;

        kd.gws0 = (output_size.feature[0] * batch_size) / split;
        kd.lws0 = std::min(kd.gws0, static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0)
        {
            kd.lws0--;
        }
        kd.gws1 = output_size.spatial[0];
        kd.gws2 = output_size.spatial[1];
        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.ofm_per_work_item = 1;
        kd.batches_per_work_item = 1;
        kd.block_width = 1;
        kd.block_height = 1;
        kd.prefetch = 0;
        kd.input_block_array_size = 0;
        kd.input_block_width = 0;
        kd.leftovers = 0;
        return kd;
    }

    typedef kd_selector_t<kernel_data, deconvolution_node, data_types, format::type, data_types, format::type, kd_optional_selector_t, int, neural::gpu::engine_info_internal::architectures, gpu::engine_info_internal::configurations> ks_type;
    static ks_type ks;

    deconvolution_gpu(const deconvolution_node& arg)
        : outer(arg)
        , input_layout(arg.input().get_output_layout())
        , weights_layout(arg.weights(0).get_output_layout())
        , output_layout(arg.get_output_layout())
        , _engine_info(arg.get_program().get_engine()->get_context()->get_engine_info())
        , _kernel_data(ks.get_kernel(arg,
            input_layout.data_type,
            input_layout.format,
            weights_layout.data_type,
            weights_layout.format,
            input_layout.size.batch[0],
            _engine_info.architecture,
            _engine_info.configuration))
        , _kernel(arg.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(), outer.id())
    {}

    gpu::jit_constants get_jit_constants() const
    {
        auto input_offset = outer.get_primitive()->input_offset;
        auto output_padding = outer.get_output_layout().data_padding;
        auto split = outer.get_primitive()->split();

        auto input_size = input_layout.size;
        tensor stride( 1, 1, outer.get_primitive()->stride.spatial[0], outer.get_primitive()->stride.spatial[1] );
        padding input_padding = outer.input().get_output_layout().data_padding;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("INPUT",                     input_size),
            gpu::make_jit_constant("OUTPUT",                    output_layout.size),
            gpu::make_jit_constant("STRIDE",                    stride),
            gpu::make_jit_constant("INPUT_OFFSET",              input_offset),
            // TODO: Output limit is incorrect for following cases (1. primitive used as input for two different convolutions with different padding, 2. asymmetric padding). Need to be checked and corrected.
            gpu::make_jit_constant("OUTPUT_LIMIT",              output_layout.size.add(output_padding.lower_size()).add(output_padding.upper_size())),
            gpu::make_jit_constant("INPUT_PADDING",             input_padding),
            gpu::make_jit_constant("OUTPUT_PADDING",            output_padding),
            gpu::make_jit_constant("FILTER",                    weights_layout.size),
            gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
            gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_BATCH_NUM"),
            gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM",  "FILTER_FEATURE_NUM"),
            gpu::make_jit_constant("FP16_SUPPORTED",            static_cast<int>(_engine_info.supports_fp16)),
            gpu::make_jit_constant("FP16_UNIT_USED",            static_cast<int>(_kernel_data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",                 _kernel_data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",             _kernel_data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("RELU",                      static_cast<int>(outer.get_primitive()->with_activation)),
            gpu::make_jit_constant("NEGATIVE_SLOPE",            outer.get_primitive()->activation_negative_slope),
        };

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& events, deconvolution_inst& instance) override
    {
        auto split = outer.get_primitive()->split();

        auto& input_mem = instance.input_memory();
        auto& output_mem = instance.output_memory();
        auto& filter_mem = instance.weights_memory(0);

        if (outer.get_output_layout().data_padding.filling_value() != 0.0f)
            throw std::invalid_argument("Unknown padding mode in convolution.");

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        if (input_mem.get_layout().data_type != output_mem.get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of output.");
        if (input_mem.get_layout().data_type != filter_mem.get_layout().data_type)
            throw std::invalid_argument("Memory format of input is incompatible with memory format of filter.");

        auto& kd = _kernel_data;

        std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>> tmp_events(events);

        // execute kernels
        for (decltype(split) i = 0; i < split; i++) {
            assert(kd.gws0 % kd.lws0 == 0);
            auto event = _kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem, gpu::input_mem, uint32_t>
                ({ { kd.gws0, kd.gws1, kd.gws2 },{ kd.lws0, kd.lws1, kd.lws2 } },
                    tmp_events,
                    input_mem,
                    output_mem,
                    instance.weights_memory(i), //filters
                    instance.bias_memory(i), //biases
                    i);
            tmp_events.clear();
            tmp_events.emplace_back(event);
        }
        return tmp_events.at(0);
    }

    static primitive_impl* create(const deconvolution_node& arg)
    {
        auto filter_layout = arg.weights(0).get_output_layout(); //convolution filter

        assert(arg.get_output_layout().size.feature[0] / arg.get_primitive()->split() == filter_layout.size.batch[0]); // memory::format oixy

        switch (filter_layout.fused_format())
        {
        // FP32 (float)
        case fuse(data_types::f32, format::bfyx):
        case fuse(data_types::f32, format::yxfb):
        case fuse(data_types::f16, format::bfyx):
        case fuse(data_types::f16, format::yxfb):
            break;
        default:
            throw std::runtime_error("deconvolution weights format unsupported");
        }

        return new deconvolution_gpu(arg);
    }
};

deconvolution_gpu::kernel_data default_oiyx(const deconvolution_node& arg)
{
    deconvolution_gpu::kernel_data kd = deconvolution_gpu::set_default(arg);
    kd.kernel_name = (arg.input().get_output_layout().format == cldnn::format::bfyx) ? kernel_name_bfyx_oiyx : kernel_name_yxfb_oiyx;
    return kd;
}

deconvolution_gpu::kernel_data default_yxio(const deconvolution_node& arg)
{
    deconvolution_gpu::kernel_data kd = deconvolution_gpu::set_default(arg);
    kd.kernel_name = (arg.input().get_output_layout().format == cldnn::format::bfyx) ? kernel_name_bfyx_yxio : kernel_name_yxfb_yxio;
    return kd;
}

deconvolution_gpu::ks_type deconvolution_gpu::ks = {
    { std::make_tuple(data_types::f32, format::yxfb, data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx },
    { std::make_tuple(data_types::f32, format::yxfb, data_types::f32, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio },
    { std::make_tuple(data_types::f32, format::bfyx, data_types::f32, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx },
    { std::make_tuple(data_types::f32, format::bfyx, data_types::f32, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio },
    { std::make_tuple(data_types::f16, format::yxfb, data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx },
    { std::make_tuple(data_types::f16, format::yxfb, data_types::f16, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio },
    { std::make_tuple(data_types::f16, format::bfyx, data_types::f16, format::bfyx, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_oiyx },
    { std::make_tuple(data_types::f16, format::bfyx, data_types::f16, format::yxfb, 0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), default_yxio },
};

namespace{
    struct attach {
        attach() {
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), deconvolution_gpu::create);
            implementation_map<deconvolution>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), deconvolution_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
}
