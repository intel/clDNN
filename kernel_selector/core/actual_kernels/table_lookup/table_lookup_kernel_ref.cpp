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

#include "table_lookup_kernel_ref.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector {

    ParamsKey TableLookupKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    KernelsData TableLookupKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::TABLE_LOOKUP);

        KernelData kd = KernelData::Default<TableLookupParams>(params);

        TableLookupParams& newParams = *static_cast<TableLookupParams*>(kd.params.get());
        const std::string kernel_id = GetEntryPoint(kernelName, params.layerID, options);

        std::stringstream jit;
        jit << GetBaseJit(newParams, kernel_id)
            << "#define TABLE_SIZE (" << newParams.lookupParams.tableSize << ")\n";

        if (newParams.lookupParams.tableFormat == Datatype::F16)
        {
            jit << "#define LUT_TYPE half\n";
        }
        else
        {
            jit << "#define LUT_TYPE float\n";
        }

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        const auto& out = newParams.output;
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { out.X().v, out.Y().v, out.Feature().v*out.Batch().v };
        kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.arguments = GetArgumentDesc(1, false, false);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::LOOKUP_TABLE, 0 });

        return{ kd };
    }
}