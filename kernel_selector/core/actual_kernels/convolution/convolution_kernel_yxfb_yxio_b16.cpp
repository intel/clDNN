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

#include "convolution_kernel_yxfb_yxio_b16.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b16::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::yxfb);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDilation();
        k.EnableSubGroup();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b16::SetDefault(const ConvolutionParams& arg) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        const auto filter_ofm_num = arg.weights.OFM().v;
        const auto batch_size = arg.output.Batch().v;
        const uint32_t min_lws = 16;

        if (arg.inputs[0].GetDType() == Datatype::F16)
        {
            const uint32_t min_ofm_per_wi = 16;
            const uint32_t min_batches_per_wi = 1;

            runInfo.ofmPerWorkItem = min_ofm_per_wi;
            if (batch_size % (4 * min_batches_per_wi * min_lws) == 0)
            {
                runInfo.batchesPerWorkItem = 4 * min_batches_per_wi; // USE_BLOCK_READ_2 + as_half4
            }
            else if (batch_size % (2 * min_batches_per_wi * min_lws) == 0)
            {
                runInfo.batchesPerWorkItem = 2 * min_batches_per_wi; // USE_BLOCK_READ_1 + as_half2
            }
            else
            {
                runInfo.batchesPerWorkItem = min_batches_per_wi;
            }
            
            runInfo.effiency = FORCE_PRIORITY_7;
        }
        else
        {
            runInfo.ofmPerWorkItem = 8;
            runInfo.batchesPerWorkItem = 2;
            runInfo.effiency = FORCE_PRIORITY_9;
        }

        runInfo.lws0 = min_lws;
        runInfo.gws0 = filter_ofm_num * batch_size / (runInfo.ofmPerWorkItem * runInfo.batchesPerWorkItem);
        
        return runInfo;
    }

    bool ConvolutionKernel_yxfb_yxio_b16::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }
        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);

        if (!CheckPitchForSplitOnly(params))
        {
            return false;
        }

        const auto filter_ofm_num = params.weights.OFM().v;
        const auto batch_size = params.output.Batch().v;
        const uint32_t min_lws = 16;

        const bool bInputValidated =
            (filter_ofm_num > 0) &&
            (batch_size > 0) &&
            (params.output.Feature().v == filter_ofm_num);

        if (!bInputValidated)
        {
            return false;
        }

        if (params.inputs[0].GetDType() == Datatype::F16)
        {
            const uint32_t min_ofm_per_wi = 16;
            const uint32_t min_batches_per_wi = 1;

            const bool bFilterOK = filter_ofm_num % min_ofm_per_wi == 0;            // Number of output features dividable by minimum number of output features processed inside work item.
            const bool bBatchOK = batch_size % (min_batches_per_wi * min_lws) == 0; // Batch size dividable by minimum number of batches processed when smallest local work size is used.

            if (!bFilterOK || !bBatchOK)
            {
                return false;
            }
        }
        else
        {
            if ((filter_ofm_num * batch_size) % min_lws != 0 ||
                batch_size < 32) // TODO: check why it's not supported
            {
                return false;
            }
        }

        return true;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b16::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        DispatchData runInfo = SetDefault(newParams);

        if (!CheckWorkGroups(runInfo))
        {
            // Internal Error - wrong calculation of global/local work group sizes
            return{};
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::yxio },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(newParams, runInfo);

        const auto batch_size = newParams.output.Batch().v;

        std::string kernel_name_postfix;
        if (newParams.inputs[0].GetDType() == Datatype::F32)
        {
            kernel_name_postfix = "_fp32";

            // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up by 25%
            // TODO - investigate why is this happening
            if (newParams.inputs[0].Feature().v > 4)
            {
                cldnn_jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
            }
        }
        else
        {
            kernel_name_postfix = "_fp16";
            if (batch_size >= 64)
            {
                cldnn_jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
            }
            else if (batch_size >= 32)
            {
                cldnn_jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_1", ""));
            }
        }

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName + kernel_name_postfix, jit, entry_point, true, !newParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}