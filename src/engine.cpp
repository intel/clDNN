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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"
#include "network_impl.h"
#include "gpu/ocl_toolkit.h"
#include "gpu/memory_gpu.h"

namespace cldnn
{
using gpu_toolkit_config = neural::gpu::configuration;

gpu_toolkit_config convert_configuration(const engine_configuration conf)
{
    gpu_toolkit_config result;
    result.compiler_options = conf.compiler_options;
    result.enable_profiling = conf.enable_profiling != 0;
    result.meaningful_kernels_names = conf.meaningful_kernels_names != 0;
    return result;
}

engine_impl::engine_impl(const engine_configuration& conf)
    : _configuration(conf)
    , _context(std::make_shared<gpu_toolkit>(convert_configuration(conf)))
{}

memory_impl* engine_impl::allocate_buffer(layout layout)
{
    try {
        return new neural::gpu::gpu_buffer(this, layout);
    }
    catch (const cl::Error& clErr)
    {
        switch (clErr.err())
        {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY:
        case CL_INVALID_BUFFER_SIZE:
            throw error("out of GPU resources", CLDNN_OUT_OF_RESOURCES);
        default:
            throw error("GPU buffer allocation failed", CLDNN_ERROR);
        }
    }
}

memory_impl* engine_impl::reinterpret_buffer(memory_impl* memory, layout new_layout)
{
    if (memory->get_engine() != this)
        throw error("trying to reinterpret buffer allocated by a different engine", CLDNN_ERROR);

    if (memory->get_layout() == new_layout)
        return memory;

    return new neural::gpu::gpu_buffer(this, new_layout, reinterpret_cast<neural::gpu::gpu_buffer*>(memory)->get_buffer());
}

bool engine_impl::is_the_same_buffer(memory_impl* mem1, memory_impl* mem2)
{
    if (mem1->get_engine() != this || mem2->get_engine() != this)
        return false;
    if (mem1 == mem2)
        return true;

    return (reinterpret_cast<neural::gpu::gpu_buffer*>(mem1)->get_buffer() == reinterpret_cast<neural::gpu::gpu_buffer*>(mem2)->get_buffer());
}

event_impl* engine_impl::create_user_event()
{
    return new user_event_gpu(cl::UserEvent(get_context()->context()));
}

program_impl* engine_impl::build_program(const topology_impl& topology, const build_options& options)
{
    return new program_impl(this, topology, options);
}

network_impl* engine_impl::build_network(const topology_impl& topology, const build_options& options)
{
    auto program = build_program(topology, options);
    return new network_impl(program);
}

network_impl* engine_impl::allocate_network(const program_impl* program)
{
    return new network_impl(program);
}

neural::gpu::engine_info_internal engine_impl::get_engine_info() const
{
    return _context->get_engine_info();
}

}
