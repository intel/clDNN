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
#include "gpu/ocl_user_event.h"

namespace cldnn
{
using gpu_toolkit_config = gpu::configuration;

gpu_toolkit_config convert_configuration(const engine_configuration conf)
{
    gpu_toolkit_config result;
    result.compiler_options = conf.compiler_options;
    result.enable_profiling = conf.enable_profiling != 0;
    result.meaningful_kernels_names = conf.meaningful_kernels_names != 0;
    result.dump_custom_program = conf.dump_custom_program != 0;
    result.single_kernel_name = conf.single_kernel_name;
    result.host_out_of_order = conf.enable_parallelisation != 0; //TODO: enable when barriers in driver will be fixed
    result.log = conf.engine_log;
    result.ocl_sources_dumps_dir = conf.sources_dumps_dir;
    return result;
}

engine_impl::engine_impl(const engine_configuration& conf)
    : _configuration(conf)
    , _context(gpu_toolkit::create(convert_configuration(conf)))
{}

memory_impl::ptr engine_impl::allocate_buffer(layout layout)
{
    try {
        return{ new gpu::gpu_buffer(this, layout), false };
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

memory_impl::ptr engine_impl::reinterpret_buffer(const memory_impl& memory, layout new_layout)
{
    if (memory.get_engine() != this)
        throw error("trying to reinterpret buffer allocated by a different engine", CLDNN_ERROR);

    try {
        return{ new gpu::gpu_buffer(this, new_layout, reinterpret_cast<const gpu::gpu_buffer&>(memory).get_buffer()), false };
    }
    catch (cl::Error const& err) {
        throw gpu::ocl_error(err);
    }
}

bool engine_impl::is_the_same_buffer(const memory_impl& mem1, const memory_impl& mem2)
{
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (&mem1 == &mem2)
        return true;

    return (reinterpret_cast<const gpu::gpu_buffer&>(mem1).get_buffer() == reinterpret_cast<const gpu::gpu_buffer&>(mem2).get_buffer());
}

event_impl::ptr engine_impl::create_user_event(bool set)
{
    try {
        return{ new gpu::user_event(get_context(), set), false };
    }
    catch (cl::Error const& err) {
        throw gpu::ocl_error(err);
    }
}

void engine_impl::flush_network()
{ 
    get_context()->flush();
}

program_impl::ptr engine_impl::build_program(const topology_impl& topology, const build_options& options)
{
    return{ new program_impl(*this, topology, options), false };
}

network_impl::ptr engine_impl::build_network(const topology_impl& topology, const build_options& options)
{
    return{ new network_impl(*this, topology, options), false };
}

network_impl::ptr engine_impl::allocate_network(const program_impl& program)
{
    return{ new network_impl(program), false };
}

void engine_impl::wait_for_events(std::vector<event_impl::ptr> const & events)
{
    if (!events.empty())
        _context->wait_for_events(events);
}

gpu::engine_info_internal engine_impl::get_engine_info() const
{
    return _context->get_engine_info();
}

void engine_impl::compile_program(program_impl&)
{
    //TODO: better compilation logic instead of a simple 'compile all'?
    _context->get_kernels_cache().build_all();
}

}
