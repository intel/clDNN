/*
// Copyright (c) 2018 Intel Corporation
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

#include "pass_manager.h"
#include "constants_propagator.h"
#include "program_node.h"
#include "layout_optimizer.h"

using namespace cldnn;

//ToDo remove friendship relation from  program_node and program_impl
void propagate_constants::run(program_impl &p)
{
    constants_propagator prop(&p);

    for (auto& node : p.processing_order)
        prop.visit_node(*node);

    auto&& to_replace = prop.calculate();

    //remove all nodes which are no longer relevant, i.e. nodes which:
    // 1. are constants, and
    // 2. do not have non-const user (so their data are not used during inference), and
    // 3. are not marked as outputs.
    // in case if node has either non-const user or is marked as output, it should be replace with cldnn::data rather than removed (see next loop)
    auto proc_itr = p.processing_order.begin();
    while (proc_itr != p.processing_order.end())
    {
        auto& node = (*proc_itr++);
        if (!node->is_constant())
            continue;
        if (node->has_non_const_user() || (node->is_output() && !node->is_type<data>()))
            continue;

        auto& users = node->users;
        auto& deps = node->dependencies;

        for (size_t idx = 0; idx < deps.size(); idx++)
        {
            deps.at(idx)->users.remove(node);
        }
        deps.clear();

        for (auto& usr : users) {
            auto& usr_deps = usr->dependencies;
            usr_deps.erase(std::remove(usr_deps.begin(), usr_deps.end(), node), usr_deps.end());
        }
        users.clear();

        if (!node->is_output())
        {
            auto rem = p.remove_if_dangling(*node);
            assert(rem && "Non-output constant node which has only constant users should have been removed during constants propagation pass");
            (void)rem;
        }
    }

    //replace all constant nodes which are relevant for inference (either used by non-const user or marked as output) with recomputed cldnn::data
    for (auto& cout : to_replace)
    {
        auto& id_to_replace = cout.first;

        //TODO: do not use API primitives internally and get rid of this last 'cldnn::memory' internal usage
        memory api_memory = details::memory_c_to_cpp_converter::convert(api_cast(cout.second.get()));
        //c-cpp converter does not retain since normally it is done inside API-impl layer (cldnn.cpp) so we need to do it manually
        cout.second->add_ref();

        auto const_data = std::make_shared<data>("_cldnn_const_prop_" + id_to_replace, api_memory /* <<< REMOVE ME WHEN POSSIBLE */);
        auto& new_node = p.get_or_create(const_data);
        auto& curr_node = *(p.nodes_map.at(id_to_replace));

        if (!curr_node.is_type<generic_layer>())
        {
            auto curr_node_deps = curr_node.get_dependencies();
            for (auto& dep : curr_node_deps)
            {
                auto dep_users = dep->get_users();
                for (auto& dep_user : dep_users)
                {
                    if (dep_user == &curr_node)
                        p.remove_connection(*dep, curr_node);
                }
            }
        }

        curr_node.dependencies.clear();
        //remove all constant users (as they will be either removed or replaced by cldnn::data which does not have any dependencies)
        curr_node.users.erase(
            std::remove_if(curr_node.users.begin(), curr_node.users.end(), [](program_node* node) { return node->is_constant(); }),
            curr_node.users.end()
        );
        p.replace(curr_node, new_node, false, false);
    }
}