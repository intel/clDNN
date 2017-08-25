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
#include "primitive_inst.h"
#include "network_impl.h"
#include "engine_impl.h"
#include "memory_impl.h"
#include "error_handler.h"

namespace cldnn
{
event_impl::ptr primitive_inst::execute(const std::vector<event_impl::ptr>& events)
{
    CLDNN_ERROR_BOOL(id(), "Invalid/unset input", !_has_valid_input, "Cannot execute primitive " + id() + " with invalid/unset input");

    on_execute();

    if (_deps.size() == 0)
        return _impl->execute(events, *this);

    std::vector<event_impl::ptr> dependencies;
    dependencies.reserve(_deps.size());

    for(auto& input : _deps)
    {
        dependencies.emplace_back(get_network().execute_primitive(input, events));
    }

     return _impl->execute(dependencies, *this);
}

primitive_inst::primitive_inst(network_impl& network, program_node const& node, bool allocate_buffer)
    : _network(network)
    , _node(node)
    , _impl(node.get_selected_impl())
    , _deps(network.get_primitives(node.get_dependencies()))
    , _output()
    , _output_changed(false)
{
    if (allocate_buffer)
        _output = allocate_output();
}

memory_impl::ptr primitive_inst::allocate_output()
{
    auto layout = _node.get_output_layout();
    return get_network().get_engine().allocate_buffer(layout);
}

std::string primitive_inst::generic_to_string(program_node const& node, const char* type_name)
{
    std::stringstream primitive_description;
    std::stringstream ss_inputs;
    for (size_t i = 0; i < node.get_dependencies().size(); ++i)
    {
        auto& in = node.get_dependency(i);
        ss_inputs << in.id();
        ss_inputs << ", count: " << in.get_output_layout().count();
        i != (node.get_dependencies().size() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    auto&& out_layout = node.get_output_layout();

    primitive_description << "id: " << node.id() << ", type: " << type_name <<
        "\n\tdeps count: " << node.get_dependencies().size() <<
        "\n\tdeps: " << ss_inputs.str() <<
        "\n\toutput: count: " << out_layout.count() << ",  size: " << out_layout.size <<
        "\n\toutput padding lower size: " << out_layout.data_padding.lower_size() <<
        "\n\toutput padding upper size: " << out_layout.data_padding.upper_size() << '\n';

    return primitive_description.str();
}

}
