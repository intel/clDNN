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

#include <memory>
#include "cnn_kernel_base.h"

namespace KernelSelector {

    std::string CNNKernelBase::GetBaseJit(const BaseParams& params, const std::string& kernel_id) const
    {
        std::stringstream jit;

        if (kernel_id.empty())
        {
            jit << "#define KERNEL(name) __kernel void name\n"
                << "#define FUNC(name) name\n"
                << "#define FUNC_CALL(name) name\n";
        }
        else
        {
            jit << "#define KERNEL(name) __kernel void " << kernel_id << "\n"
                << "#define FUNC(name) name##_" << kernel_id << "\n"
                << "#define FUNC_CALL(name) name##_" << kernel_id << "\n";
        }

        jit << "#define ACTIVATION_FUNCTION_" << toString(params.activationFunc) << "\n"
            << "#define TYPE_" << toString(params.inputs[0].GetDType()) << "\n"
            << "#define NL_M (" << Float2Str(params.nlParams.m) << ")\n"
            << "#define NL_N (" << Float2Str(params.nlParams.n) << ")\n"
            << "#define INPUT_OFFSET (" << params.inputs[0].GetFirstElementOffset() << ")\n"
            << "#define INPUT_VIEW_OFFSET (" << params.inputs[0].GetViewOffset() << ")\n"
            << "#define OUTPUT_OFFSET (" << params.output.GetFirstElementOffset() << ")\n"
            << "#define OUTPUT_VIEW_OFFSET (" << params.output.GetViewOffset() << ")\n";

        jit << "#define INPUT_SIZE_X (" << params.inputs[0].X().v << ")\n"
            << "#define INPUT_SIZE_Y (" << params.inputs[0].Y().v << ")\n"
            << "#define INPUT_FEATURE_NUM (" << params.inputs[0].Feature().v << ")\n"
            << "#define INPUT_BATCH (" << params.inputs[0].Batch().v << ")\n"
            << "#define INPUT_X_PITCH (" << params.inputs[0].X().pitch << ")\n"
            << "#define INPUT_Y_PITCH (" << params.inputs[0].Y().pitch << ")\n"
            << "#define INPUT_FEATURE_PITCH (" << params.inputs[0].Feature().pitch << ")\n"
            << "#define INPUT_BATCH_PITCH (" << params.inputs[0].Batch().pitch << ")\n";

        jit << "#define OUTPUT_SIZE_X (" << params.output.X().v << ")\n"
            << "#define OUTPUT_SIZE_Y (" << params.output.Y().v << ")\n"
            << "#define OUTPUT_FEATURE_NUM (" << params.output.Feature().v << ")\n"
            << "#define OUTPUT_BATCH_NUM (" << params.output.Batch().v << ")\n"
            << "#define OUTPUT_X_PITCH (" << params.output.X().pitch << ")\n"
            << "#define OUTPUT_Y_PITCH (" << params.output.Y().pitch << ")\n"
            << "#define OUTPUT_FEATURE_PITCH (" << params.output.Feature().pitch << ")\n"
            << "#define OUTPUT_BATCH_PITCH (" << params.output.Batch().pitch << ")\n";

        return jit.str();
    }

    Arguments CNNKernelBase::GetArgumentDesc(uint32_t num_of_input, bool use_weights, bool use_bias) const
    {
        Arguments args;

        for (uint32_t i = 0; i < num_of_input; i++)
        {
            args.push_back({ ArgumentDescriptor::Types::INPUT, i });
        }

        args.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });

        if (use_weights)
        {
            args.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
        }

        if (use_bias)
        {
            args.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        }

        return args;
    }

    std::shared_ptr<KernelString> CNNKernelBase::GetKernelString(std::string name, std::string jit, std::string entry_point, std::string exe_mode, std::string default_build_flags) const
    {
        std::shared_ptr<KernelString> kernel_string = std::make_shared<KernelString>();

        auto codes = db.get(name);

        if (codes.size())
        {
            kernel_string->str = codes[0];
            kernel_string->jit = jit;
            kernel_string->options = exe_mode + " " + default_build_flags;
            kernel_string->entry_point = entry_point;
            kernel_string->batch_compilation = true;
        }

        return kernel_string;
    }

    std::string CNNKernelBase::GetEntryPoint(const std::string& templateName, const std::string& layerID, const OptionalParams& options) const
    {
        std::string kernelID = layerID;

        if (kernelID.empty() || !options.meaningfulKernelsNames)
        {
            kernelID = templateName;
        }

        std::replace(kernelID.begin(), kernelID.end(), '.', '_');

        kernelID += "_" + std::to_string(UniqeID());

        return kernelID;
    }

}