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

#include "softmax_inst.h"
#include "kernel.h"
#include "implementation_map.h"

#include <algorithm>

using namespace cldnn;

namespace neural
{
// Kernel names.
static const std::string kernel_name_batches_yxfb = "softmax_gpu_continoues_yxfb";
static const std::string kernel_name_batches_bfyx = "softmax_gpu_continoues_bfyx";
static const std::string kernel_name_generic = "softmax_gpu_generic";

struct softmax_gpu : typed_primitive_impl<softmax>
{
    const softmax_node& outer;

    struct kernel_data
    {
        size_t gws0;
        size_t gws1;
        size_t lws0;

        std::string kernel_name;
        size_t items_num;
        size_t leftovers;
        size_t data_sets_count;
        size_t data_set_size;
        bool fp16_unit_used;
        bool fp16_supported;
        size_t norm_index; //which dimension (from in-memory representation) is normalized, e.g. for bfyx and softmax::normalize_f, it will be f's index == 2 (used only by naive kernel)
    } _kernel_data;
    gpu::kernel _kernel;


    softmax_gpu(const softmax_node& arg)
        : outer(arg),
        _kernel_data(set_kernel_data(outer)),
        _kernel(outer.get_program().get_engine()->get_context(), _kernel_data.kernel_name, get_jit_constants(outer, _kernel_data), outer.id())
    {}

    static kernel_data set_kernel_data(const softmax_node& outer)
    {
        auto engine_info = outer.get_program().get_engine()->get_context()->get_engine_info();

        auto input_layout  = outer.input().get_output_layout();  // input
        if (input_layout.format != format::yxfb && input_layout.format != format::bfyx)
            throw std::runtime_error("Softmax unsupported input format");

        auto const& input_size = input_layout.size;

        if (outer.is_padded() || outer.has_padded_dependency())
            throw std::runtime_error("Softmax currently does not support neither input nor output padding");

        kernel_data kd;

        kd.fp16_unit_used = input_layout.data_type == cldnn::data_types::f16;
        kd.fp16_supported = engine_info.supports_fp16 != 0;
        kd.leftovers = 0;
        kd.items_num = 0;
        kd.gws0 = 0;
        kd.gws1 = 0;
        kd.lws0 = 0;
        kd.norm_index = 0;
        kd.data_sets_count = 0;
        kd.data_set_size = 1;

        bool use_naive = false;

        switch (outer.get_primitive()->dimension)
        {
        case softmax::normalize_bfyx:   kd.data_set_size *= input_size.batch[0];
        case softmax::normalize_fyx:    kd.data_set_size *= input_size.feature[0];
        case softmax::normalize_yx:     kd.data_set_size *= input_size.spatial[1];
        case softmax::normalize_x:      kd.data_set_size *= input_size.spatial[0];
            use_naive = (input_layout.format != format::bfyx && outer.get_primitive()->dimension == softmax::normalize_x);
            kd.norm_index = (input_layout.format == format::bfyx ? 0 : 2);
            break;

        case softmax::normalize_y:
            kd.data_set_size = input_size.spatial[1];
            use_naive = (input_layout.format != format::yxfb);
            kd.norm_index = (input_layout.format == format::bfyx ? 1 : 3);
            break;

        case softmax::normalize_f:
            kd.data_set_size = input_size.feature[0];
            kd.norm_index = (input_layout.format == format::bfyx ? 2 : 1);
            use_naive = true;
            break;

        case softmax::normalize_b:
            kd.data_set_size = input_size.batch[0];
            break;
        }

        kd.data_sets_count = input_size.count() / kd.data_set_size;

        if (use_naive)
        {
            auto sizes = gpu::get_accumulated_sizes_array(input_layout);

            kd.gws0 = sizes[kd.norm_index];
            kd.gws1 = input_size.count() / (kd.gws0 * kd.data_set_size);
            kd.lws0 = 1;

            kd.items_num = kd.data_set_size;
            kd.kernel_name = kernel_name_generic;
        }
        else if (input_layout.format == format::bfyx)
        {
            //start with 1 thread per data set
            kd.gws0 = 1;
            kd.gws1 = kd.data_sets_count;
            kd.items_num = kd.data_set_size;

            // We have two units of data per work item in current implementation.
            auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
            // Combining device execution and local memory restrictions to compute maximum possible LWS.
            auto max_lws = std::min(engine_info.max_work_group_size, engine_info.max_local_mem_size / local_mem_per_wi);

            kd.lws0 = 1;
            // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
            while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
            {
                kd.lws0 *= 2;
                kd.items_num /= 2;
            }

            assert((kd.items_num + 1) * kd.lws0 >= kd.data_set_size && "More than 'lws0' items per batch remains! Lws too small?");

            kd.gws0 = kd.lws0;
            kd.leftovers = kd.data_set_size % kd.lws0;

            kd.kernel_name = kernel_name_batches_bfyx;
        }
        else if (input_layout.format == format::yxfb)
        {
            //start with 1 thread per data set
            kd.gws0 = kd.data_sets_count;
            kd.gws1 = 1;
            kd.items_num = kd.data_set_size;

            // We have two units of data per work item in current implementation.
            auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
            // Combining device execution and local memory restrictions to compute maximum possible LWS.
            auto max_lws = std::min(engine_info.max_work_group_size, engine_info.max_local_mem_size / local_mem_per_wi);

            kd.lws0 = kd.data_sets_count;
            // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
            while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
            {
                kd.lws0 *= 2;
                kd.items_num /= 2;
            }

            kd.gws0 = kd.lws0;
            kd.gws1 = 1;
            kd.leftovers = (kd.data_set_size * kd.data_sets_count) % kd.lws0;

            kd.kernel_name = kernel_name_batches_yxfb;
        }
            
        
        assert(kd.items_num > 0 && kd.lws0 && kd.gws0 > 0);
        return kd;
    }

    static gpu::jit_constants get_jit_constants(const softmax_node& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_layout = outer.input().get_output_layout();
        auto input_buffer_size = input_layout.get_buffer_size();
        auto output_layout = outer.get_output_layout();

        gpu::jit_constants jits{
            gpu::make_jit_constant("INPUT",          input_buffer_size),
            gpu::make_jit_constant("ITEMS_NUM",      data.items_num),
            gpu::make_jit_constant("LWS",            data.lws0),
            gpu::make_jit_constant("GWS",            data.gws0),
            gpu::make_jit_constant("DATA_SETS_COUNT",data.data_sets_count),
            gpu::make_jit_constant("DATA_SET_SIZE",  data.data_set_size),
            gpu::make_jit_constant("LEFTOVERS",      data.leftovers),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(data.fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_MAX",   data.fp16_unit_used ? "HALF_MAX" : "FLT_MAX"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",  data.fp16_unit_used ? "0.0h" : "0.0f"),
        };

        if (data.kernel_name == kernel_name_generic)
        {
            jits.add_constant(gpu::make_jit_constant("INPUT_STRIDE", gpu::get_accumulated_buffer_sizes_array(input_layout)[data.norm_index]));
            jits.add_constant(gpu::make_jit_constant("OUTPUT_STRIDE", gpu::get_accumulated_buffer_sizes_array(output_layout)[data.norm_index]));
        }

        return jits;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, softmax_inst& instance) override
    {
        const auto& kd    = _kernel_data;

        const auto& input_mem  = instance.input_memory();  // input
        const auto& output_mem = instance.output_memory(); // output

        assert(1 == output_mem.get_layout().size.feature.size());
        assert(1 == output_mem.get_layout().size.batch.size());

        return _kernel.run<gpu::input_mem, gpu::output_mem>({ { kd.gws0, kd.gws1 }, { kd.lws0, 1 } }, events, input_mem, output_mem);
    }

    static primitive_impl* create(const softmax_node& arg) { return new softmax_gpu(arg); };
};

namespace {
    struct attach {
        attach() {
            auto val_fw = softmax_gpu::create;
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<softmax>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() {}
    };
}
attach attach_impl;
} // namespace neural
