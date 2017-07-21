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

#pragma once

#include <cstddef>
#include "common_types.h"
#include "common_tools.h"
#include "tensor_type.h"

namespace KernelSelector
{
    using DataTensor = Tensor::DataTensor;
    using WeightsTensor = Tensor::WeightsTensor;
    using DataLayout = Tensor::DataLayout;
    using WeightsLayout = Tensor::WeightsLayout;
    using MultiDataTensor = std::vector<DataTensor>;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ParamsKey
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ParamsKey
    {
    public:
        ParamsKey()
        {
            key.restrict.raw = 0;
            key.machineInfo.raw = 0;
            key.inputLayout = 0;
            key.outputLayout = 0;
            key.weightsLayout = 0;
        }

        struct Key
        {
            union restrict_t
            {
                struct val_t
                {
                    uint32_t inputF16 : 1;
                    uint32_t inputF32 : 1;
                    uint32_t outputF16 : 1;
                    uint32_t outputF32 : 1;
                    uint32_t inputWeightsF16 : 1;
                    uint32_t inputWeightsF32 : 1;
                    uint32_t inputWeightsINT8 : 1;
                    uint32_t outputWeightsF16 : 1;
                    uint32_t outputWeightsF32 : 1;
                    uint32_t outputWeightsINT8 : 1;
                    uint32_t different_types : 1;
                    uint32_t offset : 1;
                    uint32_t pitches : 1;
                    uint32_t batching : 1;
                    uint32_t biasPerFeatureMap : 1;
                    uint32_t biasPerOutput : 1;
                    uint32_t nonBias : 1;
                    uint32_t activationAdditionalParamsAsInput : 1;

                    union dedicated_t
                    {
                        struct norm_t
                        {
                            uint32_t across : 1;
                            uint32_t within : 1;
                            uint32_t fixedKenrelDivider : 1;
                            uint32_t dynamicKenrelDivider : 1;
                        } norm;
                        struct pooling_t
                        {
                            uint32_t max : 1;
                            uint32_t avg : 1;
                            uint32_t floor : 1;
                            uint32_t ceil : 1;
                            uint32_t fixedKenrelDivider : 1;
                            uint32_t dynamicKenrelDivider : 1;
                        } pooling;
                        struct conv_t 
                        {
                            uint32_t split : 1;
                            uint32_t dilation : 1;
                        } conv;
                        struct fc_t {} fc;
                        struct lc_t {} lc;
                        struct softmax_t 
                        {
                            uint32_t dimX : 1;
                            uint32_t dimY : 1;
                            uint32_t dimFeature : 1;
                        } softmax;
                        struct concat_t
                        {
                            uint32_t axisX : 1;
                            uint32_t axisY : 1;
                            uint32_t axisFeature : 1;
                            uint32_t axisBatch : 1;
                        } concat;
                    } dedicated;
                } val;
                uint64_t raw;
            } restrict;

            union machine_info_t
            {
                struct val_t
                {
                    uint32_t subgroup : 1;
                } val;
                uint32_t raw;
            } machineInfo;

            static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

            uint32_t inputLayout;
            uint32_t outputLayout;
            uint32_t weightsLayout;
        };

        void EnableInputDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::F16:
                key.restrict.val.inputF16 = 1;
                break;
            case Datatype::F32:
                key.restrict.val.inputF32 = 1;
                break;
            default:
                break;
            }
        }

        void EnableOutputDataType(Datatype dt)
        {
            switch (dt)
            {
            case Datatype::F16:
                key.restrict.val.outputF16 = 1;
                break;
            case Datatype::F32:
                key.restrict.val.outputF32 = 1;
                break;
            default:
                break;
            }
        }

        void EnableInputWeightsType(WeightsType wt)
        {
            switch (wt)
            {
            case WeightsType::F16:
                key.restrict.val.inputWeightsF16 = 1;
                break;
            case WeightsType::F32:
                key.restrict.val.inputWeightsF32 = 1;
                break;
            case WeightsType::INT8:
                key.restrict.val.inputWeightsINT8 = 1;
                break;
            default:
                break;
            }
        }

        void EnableOutputWeightsType(WeightsType wt)
        {
            switch (wt)
            {
            case WeightsType::F16:
                key.restrict.val.outputWeightsF16 = 1;
                break;
            case WeightsType::F32:
                key.restrict.val.outputWeightsF32 = 1;
                break;
            case WeightsType::INT8:
                key.restrict.val.outputWeightsINT8 = 1;
                break;
            default:
                break;
            }
        }

        void EnableDifferentTypes()
        {
            key.restrict.val.different_types = 1;
        }

        void EnableInputLayout(DataLayout l)
        {
            key.inputLayout |= (1 << l);
        }

        void EnableAllInputLayout()
        {
            key.inputLayout = 0xffffffff;
        }

        void EnableOutputLayout(DataLayout l)
        {
            key.outputLayout |= (1 << l);
        }

        void EnableAllOutputLayout()
        {
            key.outputLayout = 0xffffffff;
        }

        void EnableWeightsLayout(WeightsLayout l)
        {
            key.weightsLayout |= (1 << l);
        }

        void EnableAllWeightsLayout()
        {
            key.weightsLayout = 0xffffffff;
        }

        void EnableTensorOffset()
        {
            key.restrict.val.offset = 1;
        }

        void EnableTensorPitches()
        {
            key.restrict.val.pitches = 1;
        }

        void EnableBatching()
        {
            key.restrict.val.batching = 1;
        }

        void EnableSubGroup()
        {
            key.machineInfo.val.subgroup = 1;
        }

        void EnableNonBiasTerm()
        {
            key.restrict.val.nonBias = 1;
        }

        void EnableBiasPerFeature()
        {
            key.restrict.val.biasPerFeatureMap = 1;
        }

        void EnableBiasPerOutput()
        {
            key.restrict.val.biasPerOutput = 1;
        }

        void EnableActivationAdditionalParamsAsInput()
        {
            key.restrict.val.activationAdditionalParamsAsInput = 1;
        }

        void EnableLRNMode(LRNMode m)
        {
            switch (m)
            {
            case LRNMode::ACROSS_CHANNEL:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case LRNMode::WITHIN_CHANNEL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void EnableNormalizeMode(NormalizeMode m)
        {
            switch (m)
            {
            case NormalizeMode::ACROSS_SPATIAL:
                key.restrict.val.dedicated.norm.across = 1;
                break;
            case NormalizeMode::WITHIN_SPATIAL:
                key.restrict.val.dedicated.norm.within = 1;
                break;
            default:
                break;
            }
        }

        void EnableLRNKernelDividerMode(KernelDividerMode m)
        {
            switch (m)
            {
            case KernelDividerMode::FIXED:
                key.restrict.val.dedicated.norm.fixedKenrelDivider = 1;
                break;
            case KernelDividerMode::DYNAMIC:
                key.restrict.val.dedicated.norm.dynamicKenrelDivider = 1;
                break;
            default:
                break;
            }
        }

        void EnablePoolKernelDividerMode(KernelDividerMode m)
        {
            switch (m)
            {
            case KernelDividerMode::FIXED:
                key.restrict.val.dedicated.pooling.fixedKenrelDivider = 1;
                break;
            case KernelDividerMode::DYNAMIC:
                key.restrict.val.dedicated.pooling.dynamicKenrelDivider = 1;
                break;
            default:
                break;
            }
        }

        void EnablePoolType(PoolType t)
        {
            switch (t)
            {
            case PoolType::MAX:
                key.restrict.val.dedicated.pooling.max = 1;
                break;
            case PoolType::AVG:
                key.restrict.val.dedicated.pooling.avg = 1;
                break;
            default:
                break;
            }
        }

        void EnablePoolRemainder(PoolRemainder r)
        {
            switch (r)
            {
            case PoolRemainder::FLOOR:
                key.restrict.val.dedicated.pooling.floor = 1;
                break;
            case PoolRemainder::CEIL:
                key.restrict.val.dedicated.pooling.ceil = 1;
                break;
            default:
                break;
            }
        }

        void EnableSplitSupport()
        {
            key.restrict.val.dedicated.conv.split = 1;
        }

        void EnableDilation()
        {
            key.restrict.val.dedicated.conv.dilation = 1;
        }

        void EnableSoftmaxDim(SoftmaxDim d)
        {
            switch (d)
            {
            case SoftmaxDim::X:
                key.restrict.val.dedicated.softmax.dimX = 1;
                break;
            case SoftmaxDim::Y:
                key.restrict.val.dedicated.softmax.dimY = 1;
                break;
            case SoftmaxDim::FEATURE:
                key.restrict.val.dedicated.softmax.dimFeature = 1;
                break;
            default:
                break;
            }
        }

        void EnableConcatAxis(ConcatAxis a)
        {
            switch (a)
            {
            case ConcatAxis::X:
                key.restrict.val.dedicated.concat.axisX = 1;
                break;
            case ConcatAxis::Y:
                key.restrict.val.dedicated.concat.axisY = 1;
                break;
            case ConcatAxis::FEATURE:
                key.restrict.val.dedicated.concat.axisFeature = 1;
                break;
            case ConcatAxis::BATCH:
                key.restrict.val.dedicated.concat.axisBatch = 1;
                break;
            default:
                break;
            }
        }

        bool Support(const ParamsKey& k) const
        {
            return 
                ((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw) && // check if this kernel supports this params
                ((key.machineInfo.raw & k.key.machineInfo.raw) == key.machineInfo.raw) && // check if machine supports this kernel
                ((key.inputLayout & k.key.inputLayout) != 0 || key.inputLayout == k.key.inputLayout) &&
                ((key.outputLayout & k.key.outputLayout) != 0 || key.outputLayout == k.key.outputLayout) &&
                ((key.weightsLayout & k.key.weightsLayout) != 0 || key.weightsLayout == k.key.weightsLayout);
        }

        ParamsKey Merge(const ParamsKey& k) const
        {
            ParamsKey ret;
            ret.key.restrict.raw = key.restrict.raw | k.key.restrict.raw;
            ret.key.machineInfo.raw = key.machineInfo.raw | k.key.machineInfo.raw;
            ret.key.inputLayout = key.inputLayout | k.key.inputLayout;
            ret.key.outputLayout = key.outputLayout | k.key.outputLayout;
            ret.key.weightsLayout = key.weightsLayout | k.key.weightsLayout;
            return ret;
        }

    private:
        Key key;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Params
    {
        virtual ~Params() {}

        KernelType GetType() const { return kType; }
        virtual ParamsKey GetParamsKey() const = 0;

    protected:
        Params(KernelType kt, const std::string& id) : kType(kt), layerID(id) {}
        KernelType kType;

    public:
        std::string layerID;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BaseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct BaseParams : public Params
    {
        virtual ~BaseParams() {}

        ActivationFunction  activationFunc = ActivationFunction::NONE;
        NonLinearParams     nlParams;   // TODO: rename it to "activationAdditionalParams"
        MultiDataTensor     inputs;
        DataTensor          output;

        virtual std::string to_string() const;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;

            bool bBatching = false;
            bool bPitches = false;
            bool bOffests = false;
            bool bDifferentTypes = false;

            for (const auto& i : inputs)
            {
                k.EnableInputDataType(i.GetDType());
                k.EnableInputLayout(i.GetLayout());

                bBatching       |= (i.Batch().v > 1);
                bPitches        |= (i.PitchesDifferFromLogicalDims());
                bOffests        |= (i.GetFirstElementOffset() != 0);
                bDifferentTypes |= (i.GetDType() != output.GetDType());
            }

            k.EnableOutputDataType(output.GetDType());
            k.EnableOutputLayout(output.GetLayout());

            if (bBatching)
            {
                k.EnableBatching();
            }

            if (bPitches ||
                output.PitchesDifferFromLogicalDims())
            {
                k.EnableTensorPitches();
            }

            if (bDifferentTypes)
            {
                k.EnableDifferentTypes();
            }

            if (bOffests ||
                output.GetFirstElementOffset() != 0)
            {
                k.EnableTensorOffset();
            }

            return k;
        }

    protected:

        BaseParams(KernelType kt) : Params(kt, ""), inputs(1){}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightBiasParams : public BaseParams
    {
        WeightBiasParams(KernelType kt) : BaseParams(kt) {}

        WeightsTensor weights;
        MultiDataTensor bias;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableInputWeightsType(weights.GetDType());
            
            // not needed - can be changed by reorder params
            //k.EnableWeightsLayout(weights.layout);

            assert(bias.size() <= 1);

            if (bias.empty())
            {
                k.EnableNonBiasTerm();
            }
            else if (bias[0].GetLayout() == DataLayout::bf ||
                     bias[0].GetLayout() == DataLayout::fb)
            {
                k.EnableBiasPerFeature();
            }
            else if (bias[0].GetLayout() == output.GetLayout())
            {
                k.EnableBiasPerOutput();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionParams : public WeightBiasParams
    {
        ConvolutionParams() : WeightBiasParams(KernelType::CONVOLUTION), convParams() {}
    
        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t split = 1;
        };

        DedicatedParams convParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (convParams.split > 1)
            {
                k.EnableSplitSupport();
            }

            if (convParams.dilation.x != 1 ||
                convParams.dilation.y != 1)
            {
                k.EnableDilation();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeconvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DeconvolutionParams : public WeightBiasParams
    {
        DeconvolutionParams() : WeightBiasParams(KernelType::DECONVOLUTION), deconvParams() {}

        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t split = 1;
        };

        DedicatedParams deconvParams;

        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (deconvParams.split > 1)
            {
                k.EnableSplitSupport();
            }

            if (deconvParams.dilation.x != 1 ||
                deconvParams.dilation.y != 1)
            {
                k.EnableDilation();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNParams : public BaseParams
    {
        LRNParams() : BaseParams(KernelType::LRN), lrnParams() {}

        struct DedicatedParams
        {
            LRNMode             normMode    = LRNMode::ACROSS_CHANNEL;
            KernelDividerMode   divMode     = KernelDividerMode::DONT_CARE;
            float               alpha       = 0.f;
            float               beta        = 0.f;
            float               k           = 0.f;
            uint32_t            localSize   = 0;
        };

        DedicatedParams lrnParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableLRNMode(lrnParams.normMode);
            k.EnableLRNKernelDividerMode(lrnParams.divMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizeParams : public BaseParams
    {
        NormalizeParams() : BaseParams(KernelType::NORMALIZE), normParams() {}

        struct DedicatedParams
        {
            NormalizeMode normMode = NormalizeMode::ACROSS_SPATIAL;
            float         epsilon  = 1e-10f;
            DataTensor    scaleTable;
        };

        DedicatedParams normParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnableNormalizeMode(normParams.normMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingParams : public BaseParams
    {
        PoolingParams() : BaseParams(KernelType::POOLING), poolParams() {}

        struct DedicatedParams
        {
            PoolType            poolType        = PoolType::MAX;
            PoolRemainder       remainderAction = PoolRemainder::FLOOR;
            KernelDividerMode   divMode         = KernelDividerMode::DONT_CARE;
            uSize               poolSize;
            uSize               poolStride;
            uSize               poolPad;
        };

        DedicatedParams poolParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k = BaseParams::GetParamsKey();

            k.EnablePoolType(poolParams.poolType);
            k.EnablePoolRemainder(poolParams.remainderAction);
            k.EnablePoolKernelDividerMode(poolParams.divMode);

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingParams : public BaseParams
    {
        ROIPoolingParams() : BaseParams(KernelType::ROI_POOLING) {}

        size_t rois = 0;
        size_t pitchRoisR = 0;
        size_t pitchRoisB = 0;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingV1Params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingV1Params : public BaseParams
    {
        ROIPoolingV1Params() : BaseParams(KernelType::ROI_POOLING) {}

        struct DedicatedParams
        {
            PoolType    mode         = PoolType::MAX;
            size_t      pooledWidth  = 0;
            size_t      pooledHeight = 0;
            size_t      groupSize    = 0;
            float       spatialScale = 1.f;
        };

        DedicatedParams roiParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedParams : public WeightBiasParams
    {
        FullyConnectedParams() : WeightBiasParams(KernelType::FULLY_CONNECTED) {}

        virtual ParamsKey GetParamsKey() const
        {
            return WeightBiasParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LocallyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LocallyConnectedParams : public BaseParams
    {
        LocallyConnectedParams() : BaseParams(KernelType::LOCALLY_CONNECTED), lcParams() {}

        struct DedicatedParams
        {
            uSize filterSize;
            uSize stride;
            uSize padding;
        };

        DedicatedParams lcParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationParams : public BaseParams
    {
        ActivationParams() : BaseParams(KernelType::ACTIVATION), actParams() {}

        struct DedicatedParams
        {
            MultiDataTensor inputNlParams;
        };

        DedicatedParams actParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            if (!actParams.inputNlParams.empty())
            {
                k.EnableActivationAdditionalParamsAsInput();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftmaxParams : public BaseParams
    {
        SoftmaxParams() : BaseParams(KernelType::SOFT_MAX) {}

        struct DedicatedParams
        {
            SoftmaxDim dim = SoftmaxDim::FEATURE;
        };

        DedicatedParams smParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k = BaseParams::GetParamsKey();
            k.EnableSoftmaxDim(smParams.dim);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseParams : public BaseParams
    {
        EltwiseParams() : BaseParams(KernelType::ELTWISE), eltwiseParams() {}

        struct InputType
        {
            EltwiseInputMode mode   = EltwiseInputMode::INPUT_BUFFER;
            uint32_t         index  = 0; // for inputs/temp results;
            float            scalar = 0.f;

            static InputType Buffer(uint32_t index) 
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::INPUT_BUFFER;
                input.index = index;
                return input;
            }

            static InputType Intermediate(uint32_t index)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX;
                input.index = index;
                return input;
            }

            static InputType Scalar(float s)
            {
                EltwiseParams::InputType input;
                input.mode = EltwiseInputMode::SCALAR;
                input.scalar = s;
                return input;
            }
        };

        struct Node
        {
            std::vector<InputType> inputs;
            EltwiseMode mode;
        };

        struct DedicatedParams
        {
            std::vector<EltwiseParams::Node> operations;
            bool layoutBased = false;
        };

        DedicatedParams eltwiseParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderBaseParams : public BaseParams
    {
        ReorderBaseParams() : BaseParams(KernelType::REORDER) {}

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PermuteParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PermuteParams : public ReorderBaseParams
    {
        PermuteParams() {}
        
        struct DedicatedParams
        {
            std::vector<uint16_t> order;
        };

        DedicatedParams permuteParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderVxParams : public ReorderBaseParams
    {
        ReorderVxParams() : reorderParams() {}

        struct DedicatedParams
        {
            ReorderMode mode = ReorderMode::xyzw;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            return ReorderBaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderParams : public ReorderBaseParams
    {
        ReorderParams() : reorderParams() {}

        struct DedicatedParams
        {
            MeanSubtructMode    mode = MeanSubtructMode::NONE;
            std::vector<float>  meanValues;
            DataTensor          mean;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            return ReorderBaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderWeightsParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderWeightsParams : public Params
    {
        ReorderWeightsParams() : Params(KernelType::REORDER, ""), reorderParams() {}

        struct DedicatedParams
        {
            WeightsTensor input;
            WeightsTensor output;
        };

        DedicatedParams reorderParams;

        virtual ParamsKey GetParamsKey() const
        {
            ParamsKey k;
            const auto& input = reorderParams.input;
            const auto& output = reorderParams.output;
            k.EnableWeightsLayout(input.GetLayout());
            k.EnableWeightsLayout(output.GetLayout());
            k.EnableInputWeightsType(input.GetDType());
            k.EnableOutputWeightsType(output.GetDType());

            if (input.PitchesDifferFromLogicalDims() ||
                output.PitchesDifferFromLogicalDims())
            {
                k.EnableTensorPitches();
            }

            if (input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0)
            {
                k.EnableTensorOffset();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertParams : public BaseParams
    {
        ConvertParams() : BaseParams(KernelType::CONVERT), convertParams() {}

        struct DedicatedParams
        {
            ConvertTypes covertType = ConvertTypes::U16;
        };

        DedicatedParams convertParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TableLookupParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TableLookupParams : public BaseParams
    {
        TableLookupParams() : BaseParams(KernelType::TABLE_LOOKUP), lookupParams() {}

        struct DedicatedParams
        {
            Datatype tableFormat = Datatype::F16;
            size_t tableSize = 0;
        };

        DedicatedParams lookupParams;

        virtual ParamsKey GetParamsKey() const
        {
            return BaseParams::GetParamsKey();
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatenationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConcatenationParams : public BaseParams
    {
        ConcatenationParams() : BaseParams(KernelType::CONCATENATION), concatParams() {}

        struct DedicatedParams
        {
            ConcatAxis axis = ConcatAxis::FEATURE;
        };

        DedicatedParams concatParams;

        virtual ParamsKey GetParamsKey() const
        {
            auto k =  BaseParams::GetParamsKey();
            k.EnableConcatAxis(concatParams.axis);
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct OptionalParams
    {
        virtual ~OptionalParams() {}

        KernelType GetType() const { return kType; }

        std::vector<DataLayout> inputLayouts;
        std::vector<DataLayout> outputLayouts;
        bool bSupportSubGroupExt = false;
        uint64_t maxWorkGroupSize = 1;
        uint64_t maxLocalMemSize = 16*1024*1024;
        bool meaningfulKernelsNames = false;

        virtual ParamsKey GetSupportedKey() const
        {
            ParamsKey k;

            for (auto l : inputLayouts)
            {
                k.EnableInputLayout(l);
            }

            for (auto l : outputLayouts)
            {
                k.EnableOutputLayout(l);
            }

            if (bSupportSubGroupExt)
            {
                k.EnableSubGroup();
            }

            return k;
        }

    protected:
        OptionalParams(KernelType kt) : kType(kt) {}
        KernelType kType;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightsBiasOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightsBiasOptionalParams : OptionalParams
    {
        bool allowWeightsReorder = true;
    protected:
        WeightsBiasOptionalParams(KernelType kt) : OptionalParams(kt) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionOptionalParams : WeightsBiasOptionalParams
    {
        ConvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::CONVOLUTION) {}
        bool allowPadding = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DeconvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct DeconvolutionOptionalParams : WeightsBiasOptionalParams
    {
        DeconvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::DECONVOLUTION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNOptionalParams : OptionalParams
    {
        LRNOptionalParams() : OptionalParams(KernelType::LRN) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NormalizeOptionalParams : OptionalParams
    {
        NormalizeOptionalParams() : OptionalParams(KernelType::NORMALIZE) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingOptionalParams : OptionalParams
    {
        PoolingOptionalParams() : OptionalParams(KernelType::POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingOptionalParams : OptionalParams
    {
        ROIPoolingOptionalParams() : OptionalParams(KernelType::ROI_POOLING) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedOptionalParams : WeightsBiasOptionalParams
    {
        FullyConnectedOptionalParams() : WeightsBiasOptionalParams(KernelType::FULLY_CONNECTED) {}
        bool allowReorderInput = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LocallyConnectedOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LocallyConnectedOptionalParams : OptionalParams
    {
        LocallyConnectedOptionalParams() : OptionalParams(KernelType::LOCALLY_CONNECTED) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationOptionalParams : OptionalParams
    {
        ActivationOptionalParams() : OptionalParams(KernelType::ACTIVATION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftmaxOptionalParams : OptionalParams
    {
        SoftmaxOptionalParams() : OptionalParams(KernelType::SOFT_MAX) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseOptionalParams : OptionalParams
    {
        EltwiseOptionalParams() : OptionalParams(KernelType::ELTWISE) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TableLookupOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TableLookupOptionalParams : OptionalParams
    {
        TableLookupOptionalParams() : OptionalParams(KernelType::TABLE_LOOKUP) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderVxOptionalParams : OptionalParams
    {
        ReorderVxOptionalParams() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderVxOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderOptionalParams : OptionalParams
    {
        ReorderOptionalParams() : OptionalParams(KernelType::REORDER) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertOptionalParams : OptionalParams
    {
        ConvertOptionalParams() : OptionalParams(KernelType::CONVERT) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatenationOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConcatenationOptionalParams : OptionalParams
    {
        ConcatenationOptionalParams() : OptionalParams(KernelType::CONCATENATION) {}
    };
}
