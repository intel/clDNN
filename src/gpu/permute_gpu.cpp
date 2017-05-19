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

#include "permute_inst.h"
#include "kernel.h"
#include "kd_selector.h"
#include "implementation_map.h"


using namespace cldnn;

namespace neural
{

const std::string kernelName = "reorder_GPU";

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

struct permute_gpu : typed_primitive_impl<permute>
{
    const permute_node& outer;

    gpu::engine_info_internal _engine_info;

    struct kernel_data
    {
        size_t gws0, gws1, gws2;
        std::string kernel_name;
    } _kernel_data;
    gpu::kernel _kernel;

    static kd_selector_t<kernel_data, permute_node, kd_optional_selector_t, size_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> ks;

    permute_gpu(const permute_node& arg)
    : outer(arg)
    , _engine_info(outer.get_program().get_engine()->get_context()->get_engine_info())
    , _kernel_data(ks.get_kernel(
        outer,
        outer.input().get_output_layout().format.dimension(),
        _engine_info.architecture,
        _engine_info.configuration))
    , _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer), outer.id())
    {}


    static kernel_data set_kernel_data(const permute_node& outer)
    {
        kernel_data kd;

        auto input_layout = outer.input().get_output_layout();
        auto const& input_size_raw = input_layout.get_buffer_size().raw;
        auto order = get_calculation_order(input_layout.data_type, input_layout.format);

        kd.gws2 = input_size_raw[order[3]];
        kd.gws1 = input_size_raw[order[2]];
        kd.gws0 = 1;
        for (size_t i = 0; i < 2; i++) {
            kd.gws0 *= input_size_raw[order[i]];
        }

        return kd;
    }

    // We need to specify the output idx based on input position
    static std::string get_idx_calculation(data_types dt, format::type fmt, const std::vector<uint16_t>& permute_order)
    {
            // permute_inst with permuted order provided.
            // Permute_order vector determines new order based on vector index, and its value (new order)
            // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
            // for example permute[0] = 3 means that batch will be replaced with spatial y

            auto calc_order_fmt = get_calculation_order(dt, fmt);
            std::vector<uint32_t> new_calc_order(calc_order_fmt);

            for (size_t i = 0; i < new_calc_order.size(); i++)
            {
                auto pos = std::distance(calc_order_fmt.begin(), std::find(calc_order_fmt.begin(), calc_order_fmt.end(), i));
                new_calc_order[pos] = permute_order[i];
            }

            return "return lpad[" + std::to_string(calc_order_fmt[0]) + "] + pos[" + std::to_string(new_calc_order[0]) +
                "] + (lpad[" + std::to_string(calc_order_fmt[0]) + "] + size[" + std::to_string(new_calc_order[0]) +
                "] + upad[" + std::to_string(calc_order_fmt[0]) + "]) * (lpad[" + std::to_string(calc_order_fmt[1]) +
                "] + pos[" + std::to_string(new_calc_order[1]) + "] + (lpad[" + std::to_string(calc_order_fmt[1]) +
                "] + size[" + std::to_string(new_calc_order[1]) + "] + upad[" + std::to_string(calc_order_fmt[1]) +
                "]) * (lpad[" + std::to_string(calc_order_fmt[2]) + "] + pos[" + std::to_string(new_calc_order[2]) +
                "] + (lpad[" + std::to_string(calc_order_fmt[2]) + "] + size[" + std::to_string(new_calc_order[2]) +
                "] + upad[" + std::to_string(calc_order_fmt[2]) + "]) * (lpad[" + std::to_string(calc_order_fmt[3]) +
                "] + pos[" + std::to_string(new_calc_order[3]) + "])));";
    }

    // To read input memory linearly we need to specify the order of reading
    static std::vector<uint32_t> get_calculation_order(data_types /*dt*/, format::type fmt)
    {
        switch(fmt)
        {
        // permute_inst and optional conversion cases.
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
            throw std::invalid_argument("This format is not supported in GPU permute_inst");
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

    static gpu::jit_constants get_jit_constants(const permute_node& outer)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        auto input_layout = outer.input().get_output_layout();
        auto const& input_buffer_size = input_layout.get_buffer_size();
        auto input_dimensions = input_layout.size.batch.size() + input_layout.size.feature.size() + input_layout.size.spatial.size();

        auto output_layout = outer.get_output_layout();
        auto output_dimensions = output_layout.size.batch.size() + output_layout.size.feature.size() + output_layout.size.spatial.size();

        auto input_use_half = input_layout.data_type == cldnn::data_types::f16;
        auto output_use_half = output_layout.data_type == cldnn::data_types::f16;
        int input_output_type_cvt = input_use_half != output_use_half;
        auto lower_padding = output_layout.data_padding.lower_size();
        auto upper_padding = output_layout.data_padding.upper_size();

        if (!engine_info.supports_fp16 && (input_use_half || output_use_half))
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        if (outer.input().get_output_layout().count() != outer.get_output_layout().count())
        {
            throw std::runtime_error("Input/output elements numbers mismatch!!");
        }

        auto const& permute_order = outer.get_primitive()->permute_order;

        gpu::jit_constants mem_consts{
            gpu::make_jit_constant("DIMENSIONS", std::to_string(input_dimensions)),
            gpu::make_jit_constant("OUT_FORMAT_IMPLEMENTATION", get_idx_calculation(output_layout.data_type, output_layout.format, permute_order)),
            gpu::make_jit_constant("CALCULATION_ORDER", get_calculation_order_string(input_layout.data_type, input_layout.format)),
            gpu::make_jit_constant("SRC_TYPE", input_use_half ? "ushort" : std::string("float")),
            gpu::make_jit_constant("DEST_TYPE", output_use_half ? "ushort" : std::string("float")),
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

        return mem_consts;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, permute_inst& instance) override
    {
        auto me = this;

        auto& input_mem = instance.input_memory();
        auto& output_mem = instance.output_memory();
        const auto& kd = _kernel_data;

        return me->_kernel.run<gpu::input_mem, gpu::output_mem>(
        { { kd.gws0, kd.gws1, kd.gws2 } },
            events,
            input_mem,
            output_mem);
    }

    static primitive_impl* create(const permute_node& arg)
    {
        return new permute_gpu(arg);
    }
};

permute_gpu::kernel_data set_default(const permute_node& arg)
{
    permute_gpu::kernel_data kd = permute_gpu::set_kernel_data(arg);;
    kd.kernel_name = kernelName;
    return kd;
}

kd_selector_t<permute_gpu::kernel_data, permute_node, kd_optional_selector_t, size_t, neural::gpu::engine_info_internal::architectures, neural::gpu::engine_info_internal::configurations> permute_gpu::ks = {
    { std::make_tuple(0, gpu::engine_info_internal::architectures::GEN_UNKNOWN, gpu::engine_info_internal::configurations::GT_UNKNOWN), set_default },
};

namespace {
    struct attach {
        attach() {
            implementation_map<permute>::add({
                { cldnn::engine_types::ocl, permute_gpu::create },
            });
        }
        ~attach() {}
    };
    attach attach_impl;
}
}