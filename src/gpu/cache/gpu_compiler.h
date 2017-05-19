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

#include "cache_types.h"

namespace neural { namespace gpu { namespace cache {

/// \brief Class wrapping compile feature of kernel device compiler
/// 
struct gpu_compiler
{
    static binary_data compile(context* context, const jit& compile_options, const code& code);
};

} } }
