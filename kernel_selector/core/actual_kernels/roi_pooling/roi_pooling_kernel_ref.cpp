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

#include "roi_pooling_kernel_ref.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector {

    ParamsKey ROIPoolingKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::brfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    KernelsData ROIPoolingKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::ROI_POOLING);

        KernelData kd = KernelData::Default<ROIPoolingParams>(params);

        ROIPoolingParams& newParams = *static_cast<ROIPoolingParams*>(kd.params.get());
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define FP16_SUPPORTED 1\n"
            << "#define UNIT_TYPE " << (newParams.inputs[0].GetDType() == Datatype::F16 ? "half" : "float") << "\n"
            << "#define SRC_W (" << newParams.inputs[0].X().v << ")\n"
            << "#define SRC_H (" << newParams.inputs[0].Y().v << ")\n"
            << "#define DST_W (" << newParams.output.X().v << ")\n"
            << "#define DST_H (" << newParams.output.Y().v << ")\n"
            << "#define CHAN_NUM (" << newParams.inputs[0].Feature().v << ")\n"
            << "#define ROIS_NUM (" << newParams.rois << ")\n"
            << "#define BATCH_NUM (" << newParams.inputs[0].Batch().v << ")\n"
            << "#define PITCH_SRC_H (" << newParams.inputs[0].Y().pitch << ")\n"
            << "#define PITCH_SRC_C (" << newParams.inputs[0].Feature().pitch << ")\n"
            << "#define PITCH_SRC_B (" << newParams.inputs[0].Batch().pitch << ")\n"
            << "#define PITCH_ROI_R (" << newParams.pitchRoisR << ")\n"
            << "#define PITCH_ROI_B (" << newParams.pitchRoisB << ")\n"
            << "#define PITCH_DST_H (" << newParams.output.Y().pitch << ")\n"
            << "#define PITCH_DST_C (" << newParams.output.Feature().pitch << ")\n"
            << "#define PITCH_DST_R (" << newParams.output.ROI().pitch << ")\n"
            << "#define PITCH_DST_B (" << newParams.output.Batch().pitch << ")\n";

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { newParams.output.LogicalSize(), 1, 1 };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(2, false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}