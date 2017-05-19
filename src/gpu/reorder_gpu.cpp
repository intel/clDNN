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
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"


using namespace cldnn;

namespace neural
{

const std::string kernelName = "reorder_GPU";
const std::string kernelName_subtract = "reorder_subtract_GPU";
const std::string kernelName_subtract_values = "reorder_subtract_values_GPU";
const std::string kernel_name_1d_convert = "reorder_gpu_1d_convert";
const std::string kernel_name_1d_convert_subtract = "reorder_gpu_1d_convert_subtract";
const std::string kernel_name_1d_convert_subtract_values = "reorder_gpu_1d_convert_subtract_values";
const std::string kernel_name_reorder_padding_bfyx_f32 = "reorder_gpu_padding_bfyx_f32";

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

struct reorder_gpu : typed_primitive_impl<reorder>
{
    const reorder_node& outer;

    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        std::string kernel_name;
        bool has_mean;
        bool padding_only;
        bool is_flatten;
    } _kernel_data;
    gpu::kernel _kernel;
    gpu::kernel_execution_options _exec_options;

    static kd_selector_t<kernel_data, reorder_node, kd_optional_selector_t, size_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    reorder_gpu(const reorder_node& arg)
    : outer(arg)
    , _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info())
    , _kernel_data(ks.get_kernel(
        outer,
        outer.input().get_output_layout().format.dimension(),
        _engine_info.architecture,
        _engine_info.configuration))
    , _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    , _exec_options(get_execution_options())
    {}

    static kernel_data set_kernel_data(const reorder_node& outer)
    {
        kernel_data kd;

        auto input_layout = outer.input().get_output_layout();
        auto output_layout = outer.get_output_layout();

        kd.has_mean = outer.has_mean();
        kd.is_flatten = ((input_layout.size.batch[0] != 1 && output_layout.size.batch[0] == 1) ||
            (input_layout.size.feature[0] != 1 && output_layout.size.feature[0] == 1) ||
            (input_layout.size.spatial[0] != 1 && output_layout.size.spatial[0] == 1) ||
            (input_layout.size.spatial[1] != 1 && output_layout.size.spatial[1] == 1));
        kd.padding_only = (!kd.is_flatten) && (!kd.has_mean) && outer.get_primitive()->subtract_per_feature.empty() &&
            input_layout.format == output_layout.format &&
            input_layout.format == format::bfyx &&
            outer.get_output_layout().data_padding.lower_size().feature[0] == 0 &&
            outer.get_output_layout().data_padding.lower_size().batch[0] == 0 &&
            outer.get_output_layout().data_padding.upper_size().feature[0] == 0 &&
            outer.get_output_layout().data_padding.upper_size().batch[0] == 0;

        return kd;
    }

    // We need to specify the output idx based on input position
    static std::string get_idx_calculation(data_types dt, format::type fmt)
    {
        switch (fmt)
        {
        // reorder_inst and optional conversion cases.
        // For input formats:
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
        case format::byxf:
            return "return lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[0] + pos[0])));";
        
        case format::yxfb:
            return "return lpad[0] + pos[0] + (lpad[0] + size[0] + upad[0]) * (lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3])));";
        
        case format::fyxb:
            return "return lpad[0] + pos[0] + (lpad[0] + size[0] + upad[0]) * (lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[1] + pos[1])));";
        
        case format::bfyx:
            return "return lpad[2] + pos[2] + (lpad[2] + size[2] + upad[2]) * (lpad[3] + pos[3] + (lpad[3] + size[3] + upad[3]) * (lpad[1] + pos[1] + (lpad[1] + size[1] + upad[1]) * (lpad[0] + pos[0])));";
        
        case format::os_iyx_osv16:
            return R"__C(uint _slice_id = pos[0] / 16; \
                        uint _id_in_slice = pos[0] % 16; \
                        return _id_in_slice + 16 * (pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + _slice_id * size[1])));)__C";
        
        case format::bs_xs_xsv8_bsv8:
            if (dt == data_types::f32)
            {
                return R"__C(uint _b_slice_id = pos[0] / 8; \
                        uint _b_id_in_slice = pos[0] % 8; \
                        uint _x_slice_id = pos[2] / 8; \
                        uint _x_id_in_slice = pos[2] % 8; \
                        return _b_id_in_slice + 8 * (_x_id_in_slice + 8 * _x_slice_id + _b_slice_id * size[2]);)__C";
            }
            else
                throw std::invalid_argument("This format is not supported in GPU reorder");
        
        case format::bs_x_bsv16:
            return R"__C(uint _slice_id = pos[0] / 16; \
                        uint _id_in_slice = pos[0] % 16; \
                        return _id_in_slice + 16 * (pos[2] + size[2] * _slice_id);)__C";

        default:
            throw std::invalid_argument("This format is not supported in GPU reorder_inst");
        }
    }

    // To read input memory linearly we need to specify the order of reading
    static std::vector<uint32_t> get_calculation_order(data_types /*dt*/, format::type fmt)
    {
        switch(fmt)
        {
        // reorder_inst and optional conversion cases.
        // For input formats:
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
        case format::byxf:
            return { 1, 2, 3, 0 };
        
        case format::yxfb:
            return { 0, 1, 2, 3 };
        
        case format::bfyx:
            return { 2, 3, 1, 0 };
        
        case format::fyxb:
            return { 0, 2, 3, 1 };

        default:
            throw std::invalid_argument("This format is not supported in GPU reorder_inst");
        }
    }

    // output idx for flatten
    static std::string get_idx_calculation_flatten(data_types /*odt*/, format::type ofmt)
    {
        // Flatten cases
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
        switch (ofmt)
        {
        case format::bs_xs_xsv8_bsv8:
            return R"__C(uint _b_slice_id = pos[0] / 8; \
                        uint _b_id_in_slice = pos[0] % 8; \
                        uint _x_slice_id = (pos[2] + size[2] * (pos[3] + size[3] * pos[1])) / 8; \
                        uint _x_id_in_slice = (pos[2] + size[2] * (pos[3] + size[3] * pos[1])) % 8; \
                        return _b_id_in_slice + 8 * (_x_id_in_slice + 8 * _x_slice_id + _b_slice_id * (size[2] * size[3] * size[1]));)__C";
        
        case format::bs_x_bsv16:
            return R"__C(uint _slice_id = pos[0] / 16; \
                        uint _id_in_slice = pos[0] % 16; \
                        return _id_in_slice + 16 * (pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * _slice_id)));)__C";
        
        //equivalent to axis = 1 (feature), end_axis = -1(x) in caffe
        case format::bfyx:
            return "return pos[2] + size[2] * (pos[3] + size[3] * (pos[1] + size[1] * pos[0]));";
        
        //equivalent to axis = 0 (batch), end_axis = 2(y) in caffe
        case format::yxfb:
            return "return pos[0] + size[0] * ((pos[1] * size[2] * size[3]) + size[1] * (pos[2] + size[2] * pos[3]) / size[2]);";
        
        default:
            throw std::invalid_argument("This format is not supported in GPU reorder_inst - flatten");
        }
    }

    static std::string get_calculation_order_string(data_types dt, format::type fmt)
    {
        std::ostringstream os;
        os << "(uint[]){ ";
        for(auto i : get_calculation_order(dt, fmt))
        {
            os << i << ", ";
        }
        os << " }";
        return os.str();
    }

    static gpu::jit_constants get_jit_constants(const reorder_node& outer, const kernel_data& data)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        auto input_layout = outer.input().get_output_layout();
        auto const& input_buffer_size = input_layout.get_buffer_size();
        auto input_dimensions = input_layout.size.batch.size() + input_layout.size.feature.size() + input_layout.size.spatial.size();

        auto output_layout = outer.get_output_layout();
        auto const& output_buffer_size = output_layout.get_buffer_size();
        auto output_dimensions = output_layout.size.batch.size() + output_layout.size.feature.size() + output_layout.size.spatial.size();

        auto input_use_half = input_layout.data_type == cldnn::data_types::f16;
        auto output_use_half = output_layout.data_type == cldnn::data_types::f16;
        int input_output_type_cvt = input_use_half != output_use_half;
        auto lower_padding = output_layout.data_padding.lower_size();
        auto upper_padding = output_layout.data_padding.upper_size();

        bool needs_fp16 = (input_use_half != output_use_half || //float->half or half->float conversion require fp16 support
            (input_use_half && (data.has_mean || !outer.get_primitive()->subtract_per_feature.empty()))); //half->half with subtraction require fp16 support

        if (!engine_info.supports_fp16 && needs_fp16)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        if (outer.input().get_output_layout().count() != outer.get_output_layout().count())
        {
            throw std::runtime_error("Input/output elements numbers mismatch!!");
        }

        std::string half_type_str = "half";
        if (input_use_half && output_use_half && !needs_fp16) //half->half without subtraction (so plain reorder) can be done on shorts without explicit fp16 support
            half_type_str = "ushort";


        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_dimensions)),
            gpu::make_jit_constant("OUT_FORMAT_IMPLEMENTATION", data.is_flatten ? get_idx_calculation_flatten(output_layout.data_type, output_layout.format) : get_idx_calculation(output_layout.data_type, output_layout.format)),
            gpu::make_jit_constant("CALCULATION_ORDER", get_calculation_order_string(input_layout.data_type, input_layout.format)),
            gpu::make_jit_constant("SRC_TYPE", input_use_half ? half_type_str : std::string("float")),
            gpu::make_jit_constant("DEST_TYPE", output_use_half ? half_type_str : std::string("float")),
            gpu::make_jit_constant("SRC_DEST_TYPE_CVT", input_output_type_cvt),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(engine_info.supports_fp16))
        };
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < input_dimensions; i++)
            {
                s << static_cast<uint32_t>(input_buffer_size.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("SIZE", s.str()));
        }
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < output_dimensions; i++)
            {
                s << static_cast<uint32_t>(lower_padding.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("LOWER_PADDING", s.str()));
        }
        {
            std::stringstream s;
            s << "(uint[]){ ";
            for (uint32_t i = 0; i < output_dimensions; i++)
            {
                s << static_cast<uint32_t>(upper_padding.raw[i]) << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("UPPER_PADDING", s.str()));
        }

        if (data.padding_only)
        {
            mem_consts.add_constant(gpu::make_jit_constant("INPUT", input_buffer_size));
            mem_consts.add_constant(gpu::make_jit_constant("OUTPUT", output_buffer_size));
        }
        else if (data.has_mean)
        {
            auto mean_layout = outer.mean().get_output_layout();
            auto mean_dimensions = mean_layout.size.batch.size() + mean_layout.size.feature.size() + mean_layout.size.spatial.size();

            auto subtract_use_half = mean_layout.data_type == cldnn::data_types::f16;
            int subtract_input_type_cvt = subtract_use_half != input_use_half;

            if (!engine_info.supports_fp16 && subtract_use_half)
                throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_FORMAT_IMPLEMENTATION", get_idx_calculation(mean_layout.data_type, mean_layout.format)));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_TYPE", subtract_use_half ? std::string("half") : std::string("float")));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_SRC_TYPE_CVT", subtract_input_type_cvt));
            {
                std::stringstream s;
                s << "(uint[]){ ";
                for (uint32_t i = 0; i < mean_dimensions; i++)
                {
                    // TODO: get subtract padding from mean_subtract primitive.
                    s << 0/*static_cast<uint32_t>(padding.raw[i])*/ << ", ";
                }
                s << " }";
                mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_LOWER_PADDING", s.str()));
                mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_UPPER_PADDING", s.str()));
            }

        }
        else if (!outer.get_primitive()->subtract_per_feature.empty())
        {
            std::stringstream s;
            s << "(float[]){ ";
            for (uint32_t i = 0; i < outer.get_primitive()->subtract_per_feature.size(); i++)
            {
                s << outer.get_primitive()->subtract_per_feature[i] << ", ";
            }
            s << " }";
            mem_consts.add_constant(gpu::make_jit_constant("VALUE_TO_SUBTRACT", s.str()));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_TYPE", "float"));
            mem_consts.add_constant(gpu::make_jit_constant("SUBTRACT_SRC_TYPE_CVT", input_use_half));
        }

        return mem_consts;
    }

    gpu::kernel_execution_options get_execution_options() const
    {
        auto input_layout = outer.input().get_output_layout();
        auto const& input_buffer_size = input_layout.get_buffer_size();

        auto& input_size_raw = input_buffer_size.raw;
        auto dimensions = input_layout.size.batch.size() + input_layout.size.feature.size() + input_layout.size.spatial.size();
        auto order = get_calculation_order(input_layout.data_type, input_layout.format);
        if (dimensions != order.size()) throw std::runtime_error("reorder number of input dimensions != size of indices order");

        size_t gws_2 = input_size_raw[order[dimensions - 1]];
        size_t gws_1 = input_size_raw[order[dimensions - 2]];
        size_t gws_0 = 1;
        for (size_t i = 0; i < dimensions - 2; i++) {
            gws_0 *= input_size_raw[order[i]];
        }

        return { {gws_0, gws_1, gws_2} };
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, reorder_inst& instance) override
    {
        auto me = this;

        auto& input_mem = instance.input_memory();
        auto& output_mem = instance.output_memory();

        if (_kernel_data.has_mean)
        {
            return me->_kernel.run<gpu::input_mem, gpu::output_mem, gpu::input_mem>
                (me->_exec_options,
                    events,
                    input_mem,
                    output_mem,
                    instance.mean_memory());
        }
        else if (_kernel_data.padding_only)
        {
            gpu::kernel_execution_options exec_options{
                {
                    static_cast<size_t>(input_mem.get_layout().size.batch[0]),
                    static_cast<size_t>(input_mem.get_layout().size.feature[0]),
                    static_cast<size_t>(align_to(input_mem.get_layout().size.spatial[1], 32))
                },
                {
                    1, 1, 32
                }
            };
            return me->_kernel.run<gpu::input_mem, gpu::output_mem>
                (exec_options ,
                    events,
                    input_mem,
                    output_mem);
        }
        else
        {
            return me->_kernel.run<gpu::input_mem, gpu::output_mem>
                (me->_exec_options, events,
                    input_mem,
                    output_mem);
        }
    }

    static primitive_impl* create(const reorder_node& arg)
    {
        return new reorder_gpu(arg);
    }
};

reorder_gpu::kernel_data set_default(const reorder_node& arg)
{
    reorder_gpu::kernel_data kd = reorder_gpu::set_kernel_data(arg);

    if (kd.padding_only)
    {
        kd.kernel_name = kernel_name_reorder_padding_bfyx_f32;
    }
    else
    {
        //if we got values to subtract, then choose apropriate kernel
        if (kd.has_mean)
            kd.kernel_name = kernelName_subtract;
        else if (!arg.get_primitive()->subtract_per_feature.empty())
            kd.kernel_name = kernelName_subtract_values;
        else
            kd.kernel_name = kernelName;
    }

    return kd;
}

reorder_gpu::kernel_data set_default_dim1(const reorder_node& arg)
{
    reorder_gpu::kernel_data kd = reorder_gpu::set_kernel_data(arg);

    if (kd.has_mean)
        kd.kernel_name = kernel_name_1d_convert_subtract;
    else if (!arg.get_primitive()->subtract_per_feature.empty())
        kd.kernel_name = kernel_name_1d_convert_subtract_values;
    else
        kd.kernel_name = kernel_name_1d_convert;

    return kd;
}

kd_selector_t<reorder_gpu::kernel_data, reorder_node, kd_optional_selector_t, size_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> reorder_gpu::ks = {
    { std::make_tuple(1, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default_dim1 },
    { std::make_tuple(0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
};

namespace {
    struct attach {
        attach() {
            implementation_map<reorder>::add({
                { cldnn::engine_types::ocl, reorder_gpu::create }
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}