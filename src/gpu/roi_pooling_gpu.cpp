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
#include "implementation_map.h"
#include "roi_pooling/roi_pooling_v1_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;

static inline bool hasSingleBatchOutput(const program_node & node)
{
    const auto & batch = node.get_output_layout().size.batch;

    return batch.empty() || (batch.size() == 1 && batch[0] == 1);
}

namespace neural
{

struct roi_pooling_gpu : typed_primitive_impl<roi_pooling>
{
    const roi_pooling_node& outer;
    gpu::kernel _kernel;

    roi_pooling_gpu(const roi_pooling_node& arg, const kernel_selector::kernel_data& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernelString)
    {
        _kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, roi_pooling_inst& instance) override
    {
        gpu::kernel::kernel_arguments_data args;
        args.scalars = &_kernel_data.kernels[0].scalars;
        args.inputs = { &instance.input_memory(), &instance.rois_memory() };
        args.output = &instance.output_memory();

        return _kernel.run(_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const roi_pooling_node& arg)
    {
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& output_layout   = arg.get_output_layout();
        const auto& rois_layout     = arg.rois().get_output_layout();
        const auto& primitive       = arg.get_primitive();

        const auto padding_filling_value = output_layout.data_padding.filling_value();

        if (padding_filling_value != 0.0f) {
            throw std::logic_error("ROI pooling supports only zero padding.");
        }

        if (input_layout.format != output_layout.format) {
            throw std::invalid_argument("ROI pooling input/output data format does not match.");
        }

        auto group_sz = primitive->group_sz;
        auto in_feat = input_layout.get_buffer_size().feature[0];
        auto out_feat = output_layout.get_buffer_size().feature[0];

        if (group_sz < 0 || (group_sz && in_feat != group_sz * group_sz * out_feat)) {
            throw std::invalid_argument("group_sz must be either 0 (For RoIPooling) or satisfy ifm == ofm * group_sz * group_sz (For PSRoIPooling)");
        }

        if (!hasSingleBatchOutput(arg.input())) {
            throw std::invalid_argument("PS/ RoI Pooling doesn't support batching.");
        }
        
        auto roi_params = get_default_params<kernel_selector::roi_pooling_v1_params>(arg);
        auto roi_optional_params = get_default_optional_params<kernel_selector::roi_pooling_optional_params>(arg.get_program());
        
        const auto& out = roi_params.output;
        
        const auto roi_bfyx = convert_data_tensor(rois_layout);
        const auto roi_bf = roi_bfyx.FlattenFeatureAndSpatials();
        roi_params.inputs.push_back(roi_bf);
        roi_params.output = { out.GetDims(), out.GetDType(), kernel_selector::data_layout::brfyx, out.GetViewOffset(), out.PhysicalSize(), out.GetPaddedVal() }; // TOOD: it's an hack - cldnn doesn't support roi pooling with batching
        roi_params.roiParams.mode         = primitive->mode == pooling_mode::max ? kernel_selector::pool_type::MAX : kernel_selector::pool_type::AVG;
        roi_params.roiParams.pooledWidth  = primitive->pooled_width;
        roi_params.roiParams.pooledHeight = primitive->pooled_height;
        roi_params.roiParams.spatialScale = primitive->spatial_scale;
        roi_params.roiParams.groupSize    = group_sz;

        auto& kernel_selector = kernel_selector::roi_pooling_v1_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(roi_params, roi_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Cannot find a proper kernel for " + arg.id() +" with this arguments");
        }

        auto roi_pool = new roi_pooling_gpu(arg, best_kernels[0]);

        return roi_pool;
    }
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