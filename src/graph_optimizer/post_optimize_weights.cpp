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
#include "program_helpers.h"

//ToDo remove friendship relation from program_node and program_impl

post_optimize_weights::post_optimize_weights(layout_optimizer& lo_ref) : _lo(lo_ref) {}

void post_optimize_weights::run(program_impl &p) {
    run(p, _lo);
}

//function which prepares given primitive for weights optimization
template <typename T>
void post_optimize_weights::optimize_weights(T& node, layout_optimizer& lo, program_impl &p)
{
    auto weights_offset = node.get_primitive()->input.size();
    auto bias_offset = weights_offset + program_helpers::wrap_if_single(node.get_primitive()->weights).size();
    for (auto i = weights_offset; i < bias_offset; i++)
    {
        auto& weights = node.get_dependency(i);
        auto* impl = node.get_selected_impl().get();
        auto output_layout = node.get_output_layout();
        auto& weights_node = node.get_dependency(1);
        auto weights_layout = weights_node.get_output_layout();
        const auto weights_type = layout_optimizer::data_type::weights;

        auto reorders = lo.get_generic_layer(
            impl->_weights_reorder_params,
            weights.id(),
            weights_layout,
            weights_type);

        for (auto& reorder : reorders)
        {
            //insert new generic_layer node to topology
            p.add_intermediate(reorder.first, node, i, !reorder.second);
            //set generic_layer's node output layout and implementation
            auto& g_node = node.get_dependency(i);
            g_node.get_output_layout(false);
            g_node.selected_impl = g_node.type()->choose_impl(*(p.engine), g_node);
        }
        //set the old output layout and do not invalidate users as change of weights will not affect output layout
        node.set_output_layout(output_layout, false);
    }
}
template void post_optimize_weights::optimize_weights<convolution_node>(convolution_node& node, layout_optimizer& lo, program_impl &p);
template void post_optimize_weights::optimize_weights<deconvolution_node>(deconvolution_node& node, layout_optimizer& lo, program_impl &p);
template void post_optimize_weights::optimize_weights<fully_connected_node>(fully_connected_node& node, layout_optimizer& lo, program_impl &p);

void post_optimize_weights::run(program_impl &p, layout_optimizer& lo)
{
     for (auto& nm : p.nodes_map)
    {
        auto& prim = *nm.second;
        if (prim.type() == convolution::type_id())
        {
            optimize_weights(prim.as<convolution>(), lo, p);
        }
        else if (prim.type() == deconvolution::type_id())
        {
            optimize_weights(prim.as<deconvolution>(), lo, p);
        }
        else if (prim.type() == fully_connected::type_id())
        {
            optimize_weights(prim.as<fully_connected>(), lo, p);
        }
    }
}