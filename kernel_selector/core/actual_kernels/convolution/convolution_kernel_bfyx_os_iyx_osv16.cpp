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

#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace KernelSelector 
{

    ConvolutionKernel_bfyx_os_iyx_osv16::ConvolutionKernel_bfyx_os_iyx_osv16() : ConvolutionKernelBase("convolution_gpu_bfyx_os_iyx_osv16")
    {
        // Generate the dispatch options to the auto-tuner.
        std::vector<size_t> blockWidthSizes = { 1,2,4,5,6,8,10,12,14,16 };
        std::vector<size_t> blockHeightSizes = { 1,2,3,4,5 };
        std::vector<size_t> prefetchSizes = { 1,2,3,4,5,6,8,10 };
        std::vector<std::string> executionModes = { /*AGE_BASED ,*/ ROUND_ROBIN };
        const size_t maxBlockSize = 60;

        for (auto blockWidth : blockWidthSizes)
        {
            for (auto blockHeight : blockHeightSizes)
            {
                for (auto prefetch : prefetchSizes)
                {
                    for (auto executionMode : executionModes)
                    {
                        if (blockWidth * blockHeight <= maxBlockSize)
                        {
                            autoTuneOptions.emplace_back(blockWidth, blockHeight, prefetch, executionMode);
                        }
                    }
                }
            }
        }
    }

    ParamsKey ConvolutionKernel_bfyx_os_iyx_osv16::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableSubGroup();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDilation();
        return k;
    }

    static std::pair<size_t, size_t> get_bfyx_req_input_block_dims(
        size_t output_block_width,
        size_t output_block_height,
        const uSize& filter_size,
        const uSize& stride,
        const uSize& dilation,
        size_t sub_group_size = 16,
        size_t read_chunk_size = 8,
        size_t min_read_size = 16)
    {
        assert(output_block_width > 0 && output_block_height > 0);
        assert(stride.x > 0 && stride.y > 0);
        assert(filter_size.x > 0 && filter_size.y > 0);

        // Number of elements in X dimension needed from input to compute output block without re-reading input.
        size_t input_block_req_width = (output_block_width - 1) * stride.x + (filter_size.x - 1)*dilation.x + 1;
        // Number of elements in Y dimension needed from input to compute output block without re-reading input.
        size_t input_block_req_height = (output_block_height - 1) * stride.y + (filter_size.y - 1)*dilation.y + 1;

        // Required number of elements in X dimension rounded to nearest >= read chunk size.
        size_t input_block_read_width = std::max(RoundUp(input_block_req_width, read_chunk_size), min_read_size);
        // Number of sub-group-sized vectors of unit type needed to store input block.
        size_t input_block_array_size = CeilDiv(input_block_req_height * input_block_read_width, sub_group_size);

        return std::make_pair(input_block_array_size, input_block_read_width);
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_os_iyx_osv16::SetDefaultInternal(const ConvolutionParams& arg, const size_t blockWidth, const size_t blockHeight, const size_t prefetch) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        // Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
        constexpr size_t sub_group_size = 16;

        const auto of_maps = arg.output.Feature().v;
        const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);
        runInfo.cldnnStyle.leftovers = of_threads_per_batch - of_maps;

        const auto cp = arg.convParams;

        runInfo.effiency = FORCE_PRIORITY_3;

        if (blockWidth == 0 || blockHeight == 0 || prefetch == 0) // Default case
        {
            if (cp.stride.x == 1 && cp.stride.y == 1)
            {
                if (cp.filterSize.x == 1 && cp.filterSize.y == 1)
                {
                	runInfo.cldnnStyle.blockWidth = 16;
                	runInfo.cldnnStyle.blockHeight = 1;
                	runInfo.cldnnStyle.prefetch = 4;
                }
                //if less than 16 values is required to compute one single row of output
                //then each WI shall compute one single row to maximize reuse within SIMD subgroup (this gives very nice performance results)
                else if (arg.output.X().v + (cp.filterSize.x - 1)*cp.dilation.x < sub_group_size)
                {
                	runInfo.cldnnStyle.blockWidth = arg.output.X().v;
                	runInfo.cldnnStyle.blockHeight = 1;
                	runInfo.cldnnStyle.prefetch = 4;
                }
                else if (cp.filterSize.x < 5 && cp.filterSize.y < 5)
                {
                	runInfo.cldnnStyle.blockWidth = sub_group_size - cp.filterSize.x + 1;
                	runInfo.cldnnStyle.blockHeight = 2;
                	runInfo.cldnnStyle.prefetch = 4;
                }
                else
                {
                	runInfo.cldnnStyle.blockWidth = 4;
                	runInfo.cldnnStyle.blockHeight = 3;
                	runInfo.cldnnStyle.prefetch = 4;
                }
            }
            else if (cp.stride.x == 2 && cp.stride.y == 2)
            {
            	runInfo.cldnnStyle.blockWidth = 5;
            	runInfo.cldnnStyle.blockHeight = 4;
            	runInfo.cldnnStyle.prefetch = 4;
            }
            else
            {
            	runInfo.cldnnStyle.blockWidth = 4;
            	runInfo.cldnnStyle.blockHeight = 3;
            	runInfo.cldnnStyle.prefetch = 5;
                //run_info.effiency = FORCE_PRIORITY_7; // GEMM is better
            }
        }
        else // Auto-tuner case
        {
            runInfo.cldnnStyle.blockWidth = blockWidth;
            runInfo.cldnnStyle.blockHeight = blockHeight;
            runInfo.cldnnStyle.prefetch = prefetch;
        }     

        auto input_block_dims = get_bfyx_req_input_block_dims(
            runInfo.cldnnStyle.blockWidth, 
            runInfo.cldnnStyle.blockHeight,
            cp.filterSize,
            cp.stride,
            cp.dilation,
            sub_group_size,
            runInfo.fp16UnitUsed ? sub_group_size : sub_group_size / 2,
            sub_group_size);
        runInfo.cldnnStyle.inputBlockArraySize = input_block_dims.first;
        runInfo.cldnnStyle.inputBlockWidth = input_block_dims.second;

        runInfo.gws0 = CeilDiv(arg.output.X().v, runInfo.cldnnStyle.blockWidth);
        runInfo.gws1 = CeilDiv(arg.output.Y().v, runInfo.cldnnStyle.blockHeight);
        runInfo.gws2 = of_threads_per_batch * arg.output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = sub_group_size;

        return runInfo;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_os_iyx_osv16::SetDefault(const ConvolutionParams& arg) const
    {
        return SetDefaultInternal(arg);
    }

    bool ConvolutionKernel_bfyx_os_iyx_osv16::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o) ||
            !CovolutionCheckInput(p, o))
        {
            return false;
        }

        return true;
    }

    KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetTunedKernelsDataByIndex(const Params& params, const OptionalParams& options, const int autoTuneIndex) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        if (autoTuneIndex == -1)
        {
            return GetKernelsData(params, options);
        }

        if ((autoTuneIndex < 0) || (autoTuneIndex >= (int)autoTuneOptions.size()))
        {
            return{};
        }

        std::tuple<size_t, size_t, size_t, std::string> dispatchData = autoTuneOptions[autoTuneIndex];

        KernelData kd = GetKernelDataInternal(params, options, std::get<3>(dispatchData), std::get<0>(dispatchData), std::get<1>(dispatchData), std::get<2>(dispatchData));

        return { kd };
    }

    KernelData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelDataInternal(const Params& params, const OptionalParams& options, const std::string exeMode, const size_t blockWidth, const size_t blockHeight, const size_t prefetch) const
    {
        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        DispatchData runInfo = SetDefaultInternal(newParams, blockWidth, blockHeight, prefetch);

        kd.reorderInput = CovolutionUpdateInputParams(newParams);

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            { WeightsLayout::os_iyx_osv16 },
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(newParams, runInfo);
        cldnn_jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));
        cldnn_jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", runInfo.cldnnStyle.blockWidth));
        cldnn_jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", runInfo.cldnnStyle.blockHeight));
        cldnn_jit.AddConstant(MakeJitConstant("IN_BLOCK_ARRAY_SIZE", runInfo.cldnnStyle.inputBlockArraySize));
        cldnn_jit.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", runInfo.cldnnStyle.inputBlockWidth));
        cldnn_jit.AddConstant(MakeJitConstant("PREFETCH", runInfo.cldnnStyle.prefetch));

        if (runInfo.cldnnStyle.leftovers)
        {
            cldnn_jit.AddConstant(MakeJitConstant("LEFTOVERS", runInfo.cldnnStyle.leftovers));
        }

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, exeMode, true, !newParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return kd;
    }

    KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = GetKernelDataInternal(params, options, ROUND_ROBIN);

        return { kd };
    }

    KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsDataForAutoTune(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelsData res = {};
        int index = 0;

        for (auto autoTuneOption : autoTuneOptions)
        {
            KernelData kd = GetKernelDataInternal(params, options, std::get<3>(autoTuneOption), std::get<0>(autoTuneOption), std::get<1>(autoTuneOption), std::get<2>(autoTuneOption));
            kd.autoTuneIndex = index;
            res.emplace_back(kd);
            index++;
        }

        KernelsData defaultKds = GetKernelsData(params, options);
        res.insert(res.end(), defaultKds.begin(), defaultKds.end());

        return res;
    }
}