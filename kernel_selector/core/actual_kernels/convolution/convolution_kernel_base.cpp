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

#include "convolution_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{
    bool ConvolutionKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::CONVOLUTION ||
            o.GetType() != KernelType::CONVOLUTION)
        {
            return false;
        }

        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(o);

        bool bSupportedWeightsLayout = false;

        for (WeightsLayout l : GetSupportedWeightLayouts(params))
        {
            bSupportedWeightsLayout |= params.weights.GetLayout() == l;
        }

        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowStaticInputReordering;

        if (!bWeightsOK)
        {
            return false;
        }

        return true;
    }

    JitConstants ConvolutionKernelBase::GetJitConstants(const ConvolutionParams& params, ConvolutionKernelBase::DispatchData kd) const
    {
        std::vector<uint32_t> unrollLoopParams{
            params.convParams.filterSize.x,
            params.convParams.filterSize.y,
            (uint32_t)kd.gemmStyle.globalWorkSizeDX,
            (uint32_t)kd.gemmStyle.globalWorkSizeDY,
            (uint32_t)kd.gemmStyle.globalWorkSizeDZ,
            (uint32_t)kd.gemmStyle.subBlockDimM,
            (uint32_t)kd.gemmStyle.subBlockDimK,
            (uint32_t)kd.gemmStyle.subBlockDimN
        };

        auto loopCount = *std::max_element(unrollLoopParams.begin(), unrollLoopParams.end());

        JitConstants mem_consts = MakeConvolutionParamsJitConstants(params);
        JitConstants mem_consts_loop = MakeLoopUnrollParamsJitConstants(loopCount);
        mem_consts.Merge(mem_consts_loop);

        if (params.inputs[0].GetLayout() == DataLayout::yxfb &&
            params.weights.GetLayout() == WeightsLayout::yxio)
        {
            const auto local_work_group_size = kd.lws0;
            const auto batch_size = params.output.Batch().v;

            mem_consts.AddConstants({
                MakeJitConstant("LOCAL_WORK_GROUP_SIZE",                            local_work_group_size),
                MakeJitConstant("OFM_PER_WORK_ITEM",                                kd.cldnnStyle.ofmPerWorkItem), // how many output feature maps for a single batch will a single work item produce
                MakeJitConstant("BATCHES_PER_WORK_ITEM",                            kd.cldnnStyle.batchesPerWorkItem), // how many batches will a single work item compute
                MakeJitConstant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS",    std::max(batch_size / kd.cldnnStyle.batchesPerWorkItem / local_work_group_size, static_cast<size_t>(1))), // how many local work groups we need to compute single element for each batch
                MakeJitConstant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",           batch_size / kd.cldnnStyle.batchesPerWorkItem), // how many work items we need to compute single element for each batch
            });
        }

        return mem_consts;
    }

    bool ConvolutionKernelBase::CheckWorkGroups(const ConvolutionKernelBase::DispatchData& kd) const
    {
        if (kd.gws0 == 0 ||
            kd.gws1 == 0 ||
            kd.gws2 == 0 ||
            kd.lws0 == 0 ||
            kd.lws1 == 0 ||
            kd.lws2 == 0)
        {
            return false;
        }

        if ((kd.gws0 % kd.lws0) != 0 ||
            (kd.gws1 % kd.lws1) != 0 ||
            (kd.gws2 % kd.lws2) != 0)
        {
            return false;
        }

        return true;
    }

    namespace
    {
        bool CheckTensorForSplit(const DataTensor& t, uint32_t split)
        {
            if (t.PitchesDifferFromLogicalDims())
            {
                auto feature = t.Feature();
                auto featureIndex = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
                if (featureIndex >= 0 && featureIndex+1 < (int)DataTensor::ChannelsCount(t.GetLayout()))
                {
                    if (feature.v*split <= t.GetDims()[featureIndex+1].pitch)
                    {
                        Tensor::NDims newDims = t.GetDims();
                        newDims[featureIndex].v = feature.v*split;
                        
                        DataTensor newTensor{ newDims, t.GetDType(), t.GetLayout(), t.GetViewOffset(), t.PhysicalSize(), t.GetPaddedVal()};

                        if (newTensor.PitchesDifferFromLogicalDims() == false)
                        {
                            return true;
                        }
                    }
                }

                return false;
            }

            return true;
        }
    }

    bool ConvolutionKernelBase::CheckPitchForSplitOnly(const ConvolutionParams& params) const
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return CheckTensorForSplit(params.inputs[0], params.convParams.split);
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernelBase::SetDefault(const ConvolutionParams& params, int) const
    {
        DispatchData kd;

        const auto& out = params.output;
        kd.fp16UnitUsed = out.GetDType() == Datatype::F16;
        std::vector<size_t> global;
        if (params.output.GetLayout() == DataLayout::bfyx || params.output.GetLayout() == DataLayout::byxf)
        {
            global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        }
        else
        {
            global = { out.Feature().v*out.Batch().v, out.X().v, out.Y().v };
        }

        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        
        kd.cldnnStyle.ofmPerWorkItem = 1;
        kd.cldnnStyle.batchesPerWorkItem = 1;
        kd.cldnnStyle.blockWidth = 1;
        kd.cldnnStyle.blockHeight = 1;
        kd.cldnnStyle.prefetch = 0;
        kd.cldnnStyle.inputBlockArraySize = 0;
        kd.cldnnStyle.inputBlockWidth = 0;
        kd.cldnnStyle.leftovers = 0;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData ConvolutionKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options, const std::string exeMode, int autoTuneIndex) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        if (NeedPaddedInput())
        {
            kd.reorderInput = CovolutionUpdateInputParams(newParams);
        }
        DispatchData runInfo = SetDefault(newParams, autoTuneIndex);
        
        if (!CheckWorkGroups(runInfo))
        {
            // Internal Error - wrong calculation of global/local work group sizes
            return{};
        }

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            GetSupportedWeightLayouts(newParams),
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto finalKernelName = GetKernelName(newParams);
        auto cldnnJit = GetJitConstants(newParams, runInfo);
        auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, finalKernelName, jit, entryPoint, exeMode, true, !newParams.bias.empty(), 1, newParams.convParams.int8_quantization, newParams.convParams.output_calibration);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;
        kd.autoTuneIndex = autoTuneIndex;

        return{ kd };
    }
}
