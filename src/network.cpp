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
#include "api/CPP/mutable_data.hpp"
#include "api/CPP/input_layout.hpp"

#include "error_handler.h"
#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "condition_inst.h"
#include "kernel_selector_helper.h"
#include <algorithm>

#include "gpu/ocl_toolkit.h"

namespace cldnn
{
/*
Network_impl will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by propagate_constants opt pass).
*/
network_impl::network_impl(const program_impl& program, bool is_internal)
    : _program(&program)
    , _internal(is_internal)
{
    static std::atomic<uint32_t> id_gen{ 0 };
    if (!_internal)
    {
        net_id = ++id_gen;
    }

    allocate_primitives();
    check_names();
    build_insts_deps();
    build_exec_order();

    _program->dump_memory_pool();
}

network_impl::network_impl(engine_impl& engine, const topology_impl& topo, const build_options& options, bool is_internal)
    : network_impl(*engine.build_program(topo, options, is_internal), is_internal)
{
}

network_impl::network_impl(engine_impl& engine, const std::set<std::shared_ptr<program_node>>& nodes, const build_options& options, bool is_internal)
    : network_impl(*engine.build_program(nodes, options, is_internal), is_internal)
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

    primitive_inst = find_primitive(id);
    
    if(primitive_inst == nullptr)
        throw std::runtime_error("topology doesn't contain prmitive:" + id);

    if (primitive_inst->type() != input_layout::type_id())
    {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }

    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);

    //Wait for previous execution completion
    reset_execution(true);
    input->set_data(data);
}

void cldnn::network_impl::check_names()
{
    for (auto const& prim : _primitives)
    {
        if (find_in_internal_networks(prim.first) != nullptr)
            CLDNN_ERROR_MESSAGE("Network_impl", "Found primitive with id: " + prim.first
                + "in anotother network.");
    }
}

std::shared_ptr<primitive_inst> cldnn::network_impl::find_primitive(const primitive_id& id)
{
    std::shared_ptr<primitive_inst> ret;

    if (_primitives.find(id) != _primitives.end())
        return _primitives.at(id);

    return find_in_internal_networks(id);
}

std::shared_ptr<primitive_inst> cldnn::network_impl::find_in_internal_networks(const primitive_id& id)
{
    std::shared_ptr<primitive_inst> ret;

    for (auto const& prim : _primitives)
    {
        if (prim.second->type() == condition::type_id()) //currently only condition inst contains mini networks
        {
            auto cond_inst = std::static_pointer_cast<condition_inst>(prim.second);
            ret = cond_inst->get_net_true()->find_primitive(id);
            if (ret != nullptr)
                return ret;
            ret = cond_inst->get_net_false()->find_primitive(id);
            if (ret != nullptr)
                return ret;
        }
    }
    return nullptr;
}

void network_impl::set_learning_rate(const float lr)
{
    _learning_rate = lr;
}

float network_impl::get_learning_rate()
{
    return _learning_rate;
}

std::string network_impl::get_primitive_info(const primitive_id& id) const
{    
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

void network_impl::allocate_primitives()
{
    std::vector<std::shared_ptr<program_node>> nodes_to_allocate{};
    for (auto node : _program->get_processing_order())
    {
        nodes_to_allocate.push_back(_program->get_node_ptr(node->id()));
    }
    std::sort(nodes_to_allocate.begin(), nodes_to_allocate.end(), [](std::shared_ptr<program_node> const& lhs,
                                                                     std::shared_ptr<program_node> const& rhs)
    {
        return (lhs->get_output_layout().bytes_count() > rhs->get_output_layout().bytes_count());
    });

    for (auto const& node : nodes_to_allocate)
    {
        allocate_primitive_instance(*node);
    }
}

void network_impl::build_insts_deps()
{
    for (auto& inst : _primitives)
    {
        inst.second->build_deps();
    }
}

void network_impl::build_exec_order()
{
    for (auto& node : _program->get_processing_order())
    {
        if (!node->is_type<data>() &&
            !(node->is_type<mutable_data>() && node->get_dependencies().empty()))
        {
            add_to_exec_order(node->id());
        }
    }
}
void network_impl::add_to_exec_order(const primitive_id& id)
{
    auto inst = get_primitive(id);
    _exec_order.push_back(inst);
}

void network_impl::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    //Wait for previous execution completion
    reset_execution(false);

    for (auto& inst : _exec_order)
    {
        execute_primitive(inst, events);
    }

    for (auto& inst : _program->get_processing_order())
    {
        //Special handling for mutable data. The event should be the same as the user or dependency with highest processing_num as
        //the mutable_data can be updated when is both user or dependency.
        if (inst->is_type<mutable_data>())
        {
            decltype(_program->get_processing_order().get_processing_number(inst)) proc_num = 0;
            for (auto& user : inst->get_users())
            {
                auto user_proc_num = _program->get_processing_order().get_processing_number(user);
                if (user_proc_num > proc_num)
                {
                    _events[inst->id()] = _events[user->id()];
                    proc_num = user_proc_num;
                }
            }

            if (!inst->get_dependencies().empty())
            {
                for (auto& dep : inst->get_dependencies())
                {
                    auto dep_proc_num = _program->get_processing_order().get_processing_number(dep);
                    if (dep_proc_num > proc_num)
                    {
                        _events[inst->id()] = _events[dep->id()];
                        proc_num = dep_proc_num;
                    }
                }
            }
        }
    }

    for (auto& dout : _data_outputs) //data primitives are not executed so if they are marked as output we need to add them valid events manually
    {
        _events[dout->id()] = get_engine().create_user_event(true);
    }

    for (auto& prim : _primitives)
    {
        prim.second->reset_output_change();
    }

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

std::vector<primitive_id> network_impl::get_executed_primitive_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_exec_order.size());
    for (auto const& executed_primitive : _exec_order)
    {
        ret.push_back(executed_primitive->id());
    }
    return ret;
}

std::vector<primitive_id> network_impl::get_all_primitive_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        if(primitive.second->can_be_optimized())
            ret.push_back("_optimized_");
        else
            ret.push_back(primitive.second->id());
    return ret;
}

std::vector<primitive_id> network_impl::get_all_primitive_org_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        ret.push_back(primitive.second->org_id());
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
    bool found = (it != _events.end());
    CLDNN_ERROR_BOOL(id, "Invalid primitive call ", found, "Primitive " + id + " is tried to be executed for the second time");

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
    if (node.is_input())
        _inputs.push_back(inst);
    if (node.is_output())
    {
        _outputs.push_back(inst);
        if (node.is_type<data>())
            _data_outputs.push_back(inst);
    }
}

}
