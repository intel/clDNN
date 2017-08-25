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
#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"

#include "api/CPP/data.hpp"
#include "api/CPP/input_layout.hpp"

#include "error_handler.h"
#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "kernel_selector_helper.h"
#include <algorithm>

namespace cldnn
{

network_impl::network_impl(const program_impl& program)
    : _program(&program)
{
    for (auto const& node : _program->get_nodes())
        allocate_primitive_instance(*node);

    //sanity check
    //TODO: fix run single layer design (in run single layer multiple networks can be run - i.e. weights optimization, we should have better logic to decide whether to filter or not some networks/nodes)
    /*if (get_engine()->get_context()->enabled_single_kernel()
        && (_exec_order.size() != 1 || _exec_order.front()->id() != get_engine()->get_context()->single_kernel_name()))
        throw std::runtime_error("Network allocation: layer specified to be run with 'single_kernel_name' option could not be found (layer id: '"
            + get_engine()->get_context()->single_kernel_name() + "').");*/
}

network_impl::network_impl(engine_impl& engine, const topology_impl& topo, const build_options& options)
    : network_impl(*engine.build_program(topo, options))
{
}

void network_impl::reset_execution(bool wait)
{
    if (wait && _events.size() > 0)
    {
        std::vector<event_impl::ptr> events;
        for (auto& pair : _events)
        {
            auto& ev = pair.second;
            if (ev->is_set())
                continue;

            events.push_back(ev);
        }

        get_engine().wait_for_events(events);
    }
    _events.clear();
}

void network_impl::set_input_data(const primitive_id& id, memory_impl& data)
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
    {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }

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

    for(auto& inst : _exec_order)
        execute_primitive(inst, events);

    for (auto& prim : _primitives)
        prim.second->reset_output_change();

    // Using output of previouse network as input to another one may cause hazard (in OOOQ mode) if user would not 
    // provide proper event to execution. Flushing pipeline should prevent this kind of issues. 
    // In scenarios with a big number of very small networks it can provide performance drop.
    get_engine().flush_network();
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

std::vector<std::shared_ptr<primitive_inst>> network_impl::get_primitives(const std::vector<program_node*>& nodes)
{
    std::vector<std::shared_ptr<primitive_inst>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const program_node* node) { return get_primitive(node->id()); });
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

    event_impl::ptr ev;
    if (!get_engine().get_context()->enabled_single_kernel() || get_engine().get_context()->single_kernel_name() == id)
        ev = primitive->execute(events);
    else
        ev = get_engine().create_user_event(true);

    _events.insert({ id, ev });
    return ev;
}

void network_impl::allocate_primitive_instance(program_node const& node)
{
    if (_primitives.count(node.id()))
        return;

    auto inst = node.type()->create_instance(*this, node);
    _primitives[node.id()] = inst;
    if (!node.is_type<data>())
        _exec_order.push_back(inst);
    if (node.is_input())
        _inputs.push_back(inst);
    if (node.is_output())
        _outputs.push_back(inst);
}

}
