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
#include "gpu_compiler.h"
#include "../ocl_toolkit.h"
#include <iostream>
#include <sstream>
#include <assert.h>

namespace neural { namespace gpu { namespace cache {

namespace {
    
code inject_jit(const jit& compile_options, const code& code)
{
    return compile_options + code; //TODO temporary untill we merge proper mechanism
}

}

binary_data gpu_compiler::compile(context* context, const jit& compile_options, const code& code_src) // throws cl::BuildError
{
    auto& clContext = context->context();
    code source = inject_jit(compile_options, code_src);
    cl::Program program(clContext, source, false);
	program.compile();
    auto binaries = program.getInfo<CL_PROGRAM_BINARIES>();
	assert(binaries.size() == 1 && "There should be only one binary");
	return binary_data(binaries[0].begin(), binaries[0].end());
}

} } }
