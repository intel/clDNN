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
static const std::string kernel_name              = "softmax_gpu";
static const std::string kernel_name_batches      = "softmax_gpu_batches";
static const std::string kernel_name_batches_bx   = "softmax_gpu_batches_bx";
static const std::string kernel_name_batches_bfyx = "softmax_gpu_batches_bfyx";
static const std::string kernel_name_batches_yxfb = "softmax_gpu_batches_yxfb";



struct softmax_gpu : typed_primitive_impl<softmax>
{
    const softmax_node& outer;

    struct kernel_data
    {
        size_t gws0;
        size_t gws1;
        size_t lws0;
        std::string kernel_name;
        size_t items_num, leftovers, elements_in_batch;
        bool fp16_unit_used;
        bool fp16_supported;
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
        auto const& input_buffer_size = input_layout.get_buffer_size();

        auto const& output_buffer_size = outer.get_output_layout().get_buffer_size();

        kernel_data kd;

        kd.fp16_unit_used      = input_layout.data_type == cldnn::data_types::f16;
        kd.fp16_supported      = engine_info.supports_fp16 != 0;
        auto batch_num         = output_buffer_size.batch[0];
        auto feature_num       = input_buffer_size.feature[0];
        size_t out_buffer_size = output_buffer_size.count();
        kd.leftovers = 0;
        kd.elements_in_batch = 0;
        
        if (input_buffer_size.feature[0] != 1 ||
            input_buffer_size.spatial[1] != 1)
        {
            kd.elements_in_batch = input_buffer_size.spatial[0] * input_buffer_size.spatial[1];
            kd.gws0 = cldnn::align_to(kd.elements_in_batch, 32);
            kd.gws1 = batch_num;
            kd.lws0 = 32;
            kd.items_num = feature_num;
            kd.kernel_name = (input_layout.format == cldnn::format::bfyx) ? kernel_name_batches_bfyx : kernel_name_batches_yxfb;
        }
        else if (batch_num <= 1)
        {
            kd.lws0 = std::min(std::max(out_buffer_size, static_cast<size_t>(1)), static_cast<size_t>(32));
            kd.leftovers = out_buffer_size % kd.lws0;
            kd.gws0 = out_buffer_size - kd.leftovers;
            kd.gws1 = 1;
            kd.items_num = kd.gws0 / kd.lws0;

            kd.kernel_name = kernel_name;
        }
        else if (input_layout.format == format::bfyx)
        {
            // We have two units of data per work item in current implementation.
            auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
            // Combining device execution and local memory restrictions to compute maximum possible LWS.
            auto max_lws = std::min(engine_info.max_work_group_size, engine_info.max_local_mem_size / local_mem_per_wi);

            kd.lws0 = 1;
            kd.items_num = out_buffer_size / batch_num;
            // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
            while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
            {
                kd.lws0 *= 2;
                kd.items_num /= 2;
            }

            assert((kd.items_num + 1) * kd.lws0 >= (out_buffer_size / batch_num) && "More than 'lws0' items per batch remains! Lws too small?");
            
            kd.gws0 = kd.lws0;
            kd.gws1 = batch_num;
            kd.leftovers = (out_buffer_size / batch_num) % kd.lws0;

            kd.kernel_name = kernel_name_batches_bx;
        }
        else
        {
            // We have two units of data per work item in current implementation.
            auto local_mem_per_wi = 2 * (kd.fp16_unit_used ? sizeof(half_t) : sizeof(float));
            // Combining device execution and local memory restrictions to compute maximum possible LWS.
            auto max_lws = std::min(engine_info.max_work_group_size, engine_info.max_local_mem_size / local_mem_per_wi);

            kd.lws0 = batch_num;
            kd.items_num = out_buffer_size / kd.lws0;
            // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads.
            while ((kd.items_num > 32 || kd.lws0 < kd.items_num) && (2 * kd.lws0 <= max_lws))
            {
                kd.lws0 *= 2;
                kd.items_num /= 2;
            }

            kd.gws0 = kd.lws0;
            kd.gws1 = 1;
            kd.leftovers = out_buffer_size % kd.lws0;

            kd.kernel_name = kernel_name_batches;
        }

        assert(kd.items_num > 0 && kd.lws0 && kd.gws0 > 0);
        return kd;
    }

    static gpu::jit_constants get_jit_constants(const softmax_node& outer, const kernel_data& data)
    {
        if (!data.fp16_supported && data.fp16_unit_used)
            throw std::invalid_argument("GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)");

        auto input_buffer_size = outer.input().get_output_layout().get_buffer_size();

        //kernel relies on INPUT_SIZE_X being a number of values per batch, for bfyx format, when spatials == 1,1
        //and actual number of values is stored as fueatures count (squeezenet), swap feature[0] with spatial[0]
        if (input_buffer_size.feature[0] > 1)
            input_buffer_size = tensor( input_buffer_size.batch[0], input_buffer_size.spatial[0], input_buffer_size.feature[0], input_buffer_size.spatial[1] );

        return gpu::jit_constants{
            gpu::make_jit_constant("INPUT",          input_buffer_size),
            gpu::make_jit_constant("ITEMS_NUM",      data.items_num),
            gpu::make_jit_constant("LWS",            data.lws0),
            gpu::make_jit_constant("GWS",            data.gws0),
            gpu::make_jit_constant("LEFTOVERS",      data.leftovers),
            gpu::make_jit_constant("FP16_SUPPORTED", static_cast<int>(data.fp16_supported)),
            gpu::make_jit_constant("FP16_UNIT_USED", static_cast<int>(data.fp16_unit_used)),
            gpu::make_jit_constant("UNIT_TYPE",      data.fp16_unit_used ? "half" : "float"),
            gpu::make_jit_constant("UNIT_VAL_MAX",   data.fp16_unit_used ? "HALF_MAX" : "FLT_MAX"),
            gpu::make_jit_constant("UNIT_VAL_ZERO",  data.fp16_unit_used ? "0.0h" : "0.0f"),
            gpu::make_jit_constant("ELEMENTS_NUM",   data.elements_in_batch)
        };
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
