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
#include "kernel_manager.h"

namespace neural { namespace gpu { namespace manager {

gpu_program manager::kernel_manager::get(context* context, const std::vector<std::pair<jit, primitive_id>>& primitives)
{
    std::vector<cache::binary_data> kernels;
    for (const auto& p : primitives) { kernels.push_back(selector.get(context, p.first, p.second)); }
    return gpu_linker::link(context, kernels);
}

} } }
