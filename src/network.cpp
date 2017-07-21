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
#include "network_impl.h"

#include "api/CPP/input_layout.hpp"

#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"

#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "kernel_selector_helper.h"
#include "gpu/kernel.h"

#include <algorithm>

namespace cldnn
{

network_impl::network_impl(program_impl::cptr program)
    : _program(program)
{
    for (auto const& node : _program->get_nodes())
        allocate_primitive_instance(*node);
    
    kernel_selector::cl_kernel_data cl_kernel;
    cl_kernel.kernelString = std::make_shared<kernel_selector::kernel_string>();
    cl_kernel.kernelString->str = R"(
        __kernel void warm_up_gpu(int c, int a, int b, __global int* out)
        {
            int res = (get_global_id(0) * a + get_global_id(1)) * b + get_global_id(2);
            if(a >> 3)
                res += get_local_id(1);
            if(c)
                out[get_local_id(0)] = res;
        }
    )";

    cl_kernel.kernelString->entry_point = "warm_up_gpu";
    cl_kernel.workGroups.global = { 1024, 8 };
    cl_kernel.scalars = {
        { kernel_selector::kernel_scalar_argument_types::INT32, (int32_t)0 },
        { kernel_selector::kernel_scalar_argument_types::INT32, (int32_t)111 },
        { kernel_selector::kernel_scalar_argument_types::INT32, (int32_t)7 }
    };
    cl_kernel.arguments = {
        { kernel_selector::kernel_argument_types::SCALAR, 0 },
        { kernel_selector::kernel_argument_types::SCALAR, 1 },
        { kernel_selector::kernel_argument_types::SCALAR, 2 },
        { kernel_selector::kernel_argument_types::OUTPUT, 0 },
    };

    auto eimpl      = _program->get_engine().get();
    auto context    = eimpl->get_context();
    auto api_engine = *reinterpret_cast<engine*>(&eimpl);
    auto out        = memory::allocate(api_engine, { data_types::f32 ,format::bfyx,{ 1,1,1,1 } });

    //pre-compile program and warm-up
    neural::gpu::kernel warmup_kernel(context, cl_kernel.kernelString);
    neural::gpu::kernel::kernel_arguments_data args;
    args.output = &out;
    args.scalars = &cl_kernel.scalars;

    warmup_kernel.run(cl_kernel, {}, args);

    context->queue().finish();
}

network_impl::network_impl(engine_impl::ptr engine, const topology_impl& topo, const build_options& options)
    : network_impl(program_impl::cptr(engine->build_program(topo, options), false))
{
}

void network_impl::reset_execution(bool wait)
{
    if (wait && _events.size() > 0)
    {
        std::vector<cl::Event> events;
        events.reserve(_events.size());
        for (auto& pair : _events)
        {
            auto clevent = pair.second->get();
            if (clevent.get() != CL_NONE)
            {
                auto event_status = clevent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();

                if (event_status != CL_COMPLETE)
                {
                    events.emplace_back(clevent);
                }
            }
        }
        if (events.size() > 0)
        {
            cl::WaitForEvents(events);
        }
    }
    _events.clear();
}

void network_impl::set_input_data(const primitive_id& id, memory_impl* data)
{
    std::shared_ptr<primitive_inst> primitive_inst;
    try {
        primitive_inst = _primitives.at(id);
    }
    catch (...)
    {
        throw std::runtime_error("topology doesn't contain prmitive:" + id);
    }
    if (primitive_inst->type() != input_layout::type_id())
        throw std::invalid_argument("primitive " + id + " is not an input");

    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);

    //Wait for previous execution completion
    reset_execution(true);
        
    input->set_data(data);
}

std::string network_impl::get_primitive_info(const primitive_id& id) const
{    
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

void network_impl::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    //Wait for previous execution completion
    reset_execution(false);

    for(auto& output : _outputs)
    {
        auto output_event = execute_primitive(output, events);
    }

    for (auto& prim : _primitives)
        prim.second->reset_output_change();
}

std::vector<primitive_id> network_impl::get_output_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_outputs.size());
    for (auto const& output : _outputs)
        ret.push_back(output->id());
    return ret;
}

std::shared_ptr<primitive_inst> network_impl::get_primitive(const primitive_id& id)
{
    if (!_primitives.count(id))
        allocate_primitive_instance(_program->get_node(id));

    return _primitives.at(id);
}

std::vector<std::shared_ptr<primitive_inst>> network_impl::get_primitives(const std::vector<primitive_id>& ids)
{
    std::vector<std::shared_ptr<primitive_inst>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) { return get_primitive(id); });
    return result;
}

refcounted_obj_ptr<event_impl> network_impl::execute_primitive(const std::shared_ptr<primitive_inst>& primitive, const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    auto id = primitive->id();
    auto it = _events.find(id);
    if(it != _events.end())
    {
        return it->second;
    }
    return _events.insert({ id, primitive->execute(events) }).first->second;
}

void network_impl::allocate_primitive_instance(program_node const& node)
{
    if (_primitives.count(node.id()))
        return;

    auto inst = node.type()->create_instance(*this, node);
    _primitives[node.id()] = inst;
    if (node.is_input())
        _inputs.push_back(inst);
    if (node.is_output())
        _outputs.push_back(inst);
}

}
