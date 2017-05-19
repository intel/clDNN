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

#include "primitive_selector.h"
#include "gpu_linker.h"

namespace neural { namespace gpu { namespace manager {

/// \brief Class building gpu programs out of best available 
/// kernels for a list of primitives accompaniated by jit
///
struct kernel_manager
{
    kernel_manager( ) = default;
    gpu_program get(context* context, const std::vector<std::pair<jit, primitive_id>>& primitives);

private:
    primitive_selector selector;
};

} } }
