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

#include "eltwise_kernel_base.h"
#include "kernel_selector_utils.h" 

namespace KernelSelector
{
    static uint32_t GetNumberOfInputs(EltwiseMode m)
    {
        switch (m)
        {
        case EltwiseMode::ADD:
        case EltwiseMode::SUB:
        case EltwiseMode::MUL:
        case EltwiseMode::DIV:
        case EltwiseMode::MIN:
        case EltwiseMode::MAX:
        case EltwiseMode::POW:
        case EltwiseMode::MODULU:
            return 2;
        case EltwiseMode::SQRT:
        case EltwiseMode::RSQRT:
        case EltwiseMode::ASSIGN:
            return 1;
        default:
            return 0;
        }
    }

    bool EltwiseKernelBase::Validate(const Params& p, const OptionalParams& o) const
    {
        if (p.GetType() != KernelType::ELTWISE ||
            o.GetType() != KernelType::ELTWISE)
        {
            return false;
        }

        const EltwiseParams& params = static_cast<const EltwiseParams&>(p);

        if (params.inputs.size() == 0)
        {
            return false;
        }

        auto& operations = params.eltwiseParams.operations;

        if (operations.size() == 0)
        {
            return false;
        }

        for (size_t op_num = 0; op_num < operations.size(); op_num++)
        {
            const auto& ew = operations[op_num];

            if (ew.inputs.size() != GetNumberOfInputs(ew.mode))
            {
                return false;
            }

            for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++)
            {
                const auto& input = ew.inputs[input_idx];
                if (input.mode == EltwiseInputMode::INPUT_BUFFER &&
                    input.index >= params.inputs.size())
                {
                    return false;
                }
            }
        }

        return true;
    }

    JitConstants EltwiseKernelBase::GetJitConstantsCommon(const EltwiseParams& params, bool useVload8) const
    {
        auto jit = MakeEltwiseJitConstants(params);

        std::string inputs_decls, vload_decls;
        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            inputs_decls += "const __global " + toCLType(params.inputs[i].GetDType()) + "* input" + std::to_string(i) + ", ";
            if (useVload8)
            {
                vload_decls += "\\\n\tconst " + toCLType(params.inputs[i].GetDType()) + "8 in" + std::to_string(i);
                if (params.inputs[i].PhysicalSize() == 1) //Scalar case
                    vload_decls += " = (" + toCLType(params.inputs[i].GetDType()) + "8)(input" + std::to_string(i) + "[0]";
                else //Buffer case
                    vload_decls += " = vload8(global_id, input" + std::to_string(i);
                vload_decls += ");";
            }
        }

        jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));
        jit.AddConstant(MakeJitConstant("ELTWISE_NO_PITCH_SAME_DIMS", CheckInputsOutputNoPitchSameDims(params)));

        if (useVload8)
            jit.AddConstant(MakeJitConstant("VLOAD_DECLS", vload_decls));

        std::string do_eltwise;

        auto& operations = params.eltwiseParams.operations;

        for (size_t op_num = 0; op_num < operations.size(); op_num++)
        {
            const std::string op_num_str = std::to_string(op_num);
            const auto& ew = operations[op_num];

            for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++)
            {
                const auto& input = ew.inputs[input_idx];
                const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);
                switch (input.mode)
                {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    if(useVload8)
                        jit.AddConstant(MakeJitConstant(name, "in" + std::to_string(input.index)));
                    else
                        jit.AddConstant(MakeJitConstant(name, "input" + std::to_string(input.index) + "[GET_INDEX(INPUT, " + std::to_string(input.index) + ")]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "input" + std::to_string(input.index) + "[(size_t)tmp" + std::to_string(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + std::to_string(input.tmpIndex)));
                    break;
                default:
                    break;
                }
            }

            std::string input0_str, input1_str, cast_type, op;

            if (useVload8)
            {
                cast_type = "(MAKE_VECTOR_TYPE(UNIT_TYPE, 8))";
                op = "MAKE_VECTOR_TYPE(UNIT_TYPE, 8) tmp" + op_num_str + " = ";
            }
            else
            {
                cast_type = "(UNIT_TYPE)";
                op = "UNIT_TYPE tmp" + op_num_str + " = ";
            }

            input0_str = cast_type + "INPUT_" + op_num_str + "_0";
            input1_str = cast_type + "INPUT_" + op_num_str + "_1";

            switch (ew.mode)
            {
            case EltwiseMode::ADD:      op += input0_str + " + " + input1_str; break;
            case EltwiseMode::SUB:      op += input0_str + " - " + input1_str; break;
            case EltwiseMode::MUL:      op += input0_str + " * " + input1_str; break;
            case EltwiseMode::DIV:      op += input0_str + " / " + input1_str; break;
            case EltwiseMode::MODULU:   op += input0_str + " % " + input1_str; break;
            case EltwiseMode::MIN:      op += cast_type + "fmin(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::MAX:      op += cast_type + "fmax(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::POW:      op += cast_type + "pow(" + input0_str + ", " + input1_str + ")"; break;
            case EltwiseMode::SQRT:     op += cast_type + "sqrt(" + input0_str + ")"; break;
            case EltwiseMode::RSQRT:    op += cast_type + "1/sqrt(" + input0_str + ")"; break;
            case EltwiseMode::ASSIGN:   op += input0_str; break;
            default:
                break;;
            }

            std::string opname = "OPERATION" + op_num_str;
            jit.AddConstant(MakeJitConstant(opname, op));
            do_eltwise += "\\\n\t" + opname + ";";
        }

        do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

        jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

        if (params.eltwiseParams.layoutBased)
        {
            jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
        }

        return jit;
    }

    JitConstants EltwiseKernelBase::GetJitConstants(const EltwiseParams& params) const
    {
        return GetJitConstantsCommon(params, false);
    }

    KernelsData EltwiseKernelBase::GetCommonKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<EltwiseParams>(params);
        EltwiseParams& newParams = *static_cast<EltwiseParams*>(kd.params.get());

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        if (newParams.eltwiseParams.layoutBased)
        {
            kernel.workGroups.global = GetTensorFriendlyWorkGroups(newParams.inputs[0]);
        }
        else if (CheckInputsOutputNoPitchSameDims(newParams))
        {
            kernel.workGroups.global = { newParams.inputs[0].LogicalSize(), 1, 1 };
        }
        else
        {
            std::vector<size_t> gws;
            for (const auto& o : out.GetDims())
            {
                gws.push_back(o.v);
            }

            for (size_t i = gws.size(); i < 4; i++)
            {
                gws.push_back(1U);
            }

            kernel.workGroups.global = { gws[0], gws[1], gws[2] * gws[3] };
        }
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, ROUND_ROBIN);
        kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}