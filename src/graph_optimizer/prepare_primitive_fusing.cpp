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

#include "api/CPP/pooling.hpp"
#include "api/CPP/proposal.hpp"
#include "api/CPP/roi_pooling.hpp"

#include "program_helpers.h"
#include "pass_manager.h"

#include "activation_inst.h"
#include "batch_norm_inst.h"
#include "batch_norm_grad_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "fused_conv_bn_scale_inst.h"
#include "lrn_inst.h"
#include "mutable_data_inst.h"
#include "mvn_inst.h"
#include "normalize_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "softmax_inst.h"
#include "scale_inst.h"
#include "scale_grad_weights_inst.h"
#include "upsampling_inst.h"


void prepare_primitive_fusing::fuse_skip_layers(program_impl &p, program_node* node)
{
    program_helpers::do_for_types<eltwise>(*node, [&p](eltwise_node& node)
    {
        bool skippable = false;
        int index = 0;
        if (node.get_primitive()->mode != eltwise_mode::sum || node.inputs_count() != 2)
            return;

        if (node.input(0).is_type<deconvolution>())
        {
            skippable = true;
        }
        else if (node.input(1).is_type<deconvolution>())
        {
            skippable = true;
            index = 1;
        }

        if (!skippable)
            return;

        auto& to_fuse_with = node.input(index);
        int to_fuse_index = index == 0 ? 1 : 0;

        //remove dependencies and users of elwtise that is going to be extracted
        p.add_connection(node.input(to_fuse_index), to_fuse_with);
        p.remove_connection(node.input(to_fuse_index), node);

        //replace processing_num of the node where fusing take place and eltwise
        auto new_processing_num = node.processing_num;      //ToDo: avoid direct modifications of processing_num
        p.processing_order.erase(p.processing_order.get_processing_iterator(to_fuse_with));
        p.processing_order.insert(p.processing_order.get_processing_iterator(node), &to_fuse_with);
        to_fuse_with.processing_num = new_processing_num;   //ToDo: avoid direct modifications of processing_num

                                                            //make sure that new fused node's users have higher processing_num than fused node
        for (auto user : to_fuse_with.get_users())
        {
            if (user->processing_num < new_processing_num)
            {
                p.processing_order.erase(p.processing_order.get_processing_iterator(*user));
                p.processing_order.insert(std::next(p.processing_order.get_processing_iterator(to_fuse_with)), user);
                user->processing_num = new_processing_num + 1; //Todo: avoid direct modifications of processing_num
            }
        }

        if (node.get_fused_activation_func() != activation_none)
            to_fuse_with.set_fused_activation(node.get_fused_activation_func(), node.get_fused_activation_params());
        to_fuse_with.set_output_padding(node.get_output_layout().data_padding);

        p.extract_and_remove(node);
    });
}

template<typename T>
static bool node_is_type(program_node* n)
{
    return n->is_type<T>();
}

void prepare_primitive_fusing::fuse_conv_bn_scale(program_impl &p, program_node* node)
{
    program_helpers::do_for_types<convolution>(*node, [&p](convolution_node& node)
    {
        if (node.users.size() > 2)
            return;

        auto found_bn = std::find_if(node.users.begin(), node.users.end(), node_is_type<batch_norm>);
        auto bn_node = found_bn != node.users.end() ? *found_bn : nullptr;
        if (bn_node != nullptr)
        {
            if (bn_node->users.size() > 2)
                return;

            auto found_scale = std::find_if(bn_node->users.begin(), bn_node->users.end(), node_is_type<scale>);
            auto sc_node = found_bn != node.users.end() ? *found_scale : nullptr;
            if (sc_node != nullptr)
            {
                int bn_index = int(std::distance(node.users.begin(), found_bn));
                int sc_index = int(std::distance(bn_node->users.begin(), found_scale));
                auto scale_prim = std::static_pointer_cast<const scale>(sc_node->get_primitive());
                auto bn_prim = std::static_pointer_cast<const batch_norm>(bn_node->get_primitive());
                auto prim = node.get_primitive();
                bool training = false;

                if (node.users.size() == 2)
                {
                    training = true;
                    float zero = 0.0f;
                    layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));

                    auto bn_backw = node.users.begin();
                    std::advance(bn_backw, bn_index == 0 ? 1 : 0);
                    if (!(*bn_backw)->is_type<batch_norm_grad>())
                        return;
                    auto sc_backw = bn_node->users.begin();
                    std::advance(sc_backw, sc_index == 0 ? 1 : 0);
                    if (!(*sc_backw)->is_type<scale_grad_weights>())
                        return;

                    auto conv_out_prim = std::make_shared<mutable_data>(prim->id + "_fused_conv_out", memory::attach(dummy_layout, &zero, 1));
                    auto& conv_out_node = p.get_or_create(conv_out_prim);
                    auto conv_out_mem = p.engine->allocate_memory(node.get_output_layout());
                    conv_out_node.as<mutable_data>().attach_memory(*conv_out_mem, false);
                    p.add_intermediate(conv_out_node, **bn_backw, 1, true);

                    auto bn_out_prim = std::make_shared<mutable_data>(prim->id + "_fused_bn_out", memory::attach(dummy_layout, &zero, 1));
                    auto& bn_out_node = p.get_or_create(bn_out_prim);
                    auto bn_out_mem = p.engine->allocate_memory(bn_node->get_output_layout());
                    bn_out_node.as<mutable_data>().attach_memory(*bn_out_mem, false);
                    p.add_intermediate(bn_out_node, **sc_backw, 0, true);
                }

                auto new_conv = std::make_shared<fused_conv_bn_scale>(prim->id + "_fused", prim->input[0], prim->weights.ref(), prim->bias.ref(), bn_prim->epsilon,
                    scale_prim->input[1], scale_prim->bias, prim->stride, prim->dilation, prim->input_offset, bn_prim->inv_variance,
                    prim->with_activation, prim->activation_negative_slope, prim->output_padding);
                auto& new_node = p.get_or_create(new_conv);
                p.replace(node, new_node, false, false);

                while (sc_node->get_dependencies().size() > 1)
                {
                    auto& dep = sc_node->get_dependency(sc_node->get_dependencies().size() - 1);
                    p.remove_connection(dep, *sc_node);
                    dep.users.push_back(&new_node);
                    if (sc_node->get_dependencies().size() == 1)
                        new_node.dependencies.insert(new_node.dependencies.begin() + 1, &dep);
                    else
                        new_node.dependencies.push_back(&dep);
                }
                p.extract_and_remove(*sc_node);
                while (bn_node->get_dependencies().size() > 1)
                {
                    auto& dep = bn_node->get_dependency(bn_node->get_dependencies().size() - 1);
                    p.remove_connection(dep, *bn_node);
                    new_node.dependencies.push_back(&dep);
                }
                p.extract_and_remove(*bn_node);
                auto inv_var_node = std::find_if(new_node.dependencies.begin(), new_node.dependencies.end(),
                    [&new_conv](const program_node* node) { return node->id().find(new_conv->inv_variance) != std::string::npos; });
                (*inv_var_node)->users.push_back(&new_node);

                if (training)
                {
                    auto user = std::find_if(new_node.users.begin(), new_node.users.end(),
                        [](const program_node* node) { return node->id().find("_fused_conv_out") != std::string::npos; });
                    p.reverse_connection(new_node, **user);
                    user = std::find_if(new_node.users.begin(), new_node.users.end(), 
                        [](const program_node* node) { return node->id().find("_fused_bn_out") != std::string::npos; });
                    p.reverse_connection(new_node, **user);
                    p.processing_order.calculate_BFS_processing_order();
                }
            }
        }
    });
}

void prepare_primitive_fusing::run(program_impl &p)
{
    bool is_debug = p.options.get<build_option_type::debug>()->enabled();

    std::list<program_node*> conv_nodes;
    auto itr = p.processing_order.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != p.processing_order.end())
    {
        auto node_itr = itr++;
        if ((*node_itr)->is_type<convolution>())
            conv_nodes.push_back(*node_itr);
    }
    itr = conv_nodes.begin();
    while (itr != conv_nodes.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        fuse_conv_bn_scale(p, node);
    }

    itr = p.processing_order.begin();
    while (itr != p.processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        program_helpers::do_for_types<activation>(*node, [&p, is_debug](activation_node& node)
        {
            auto& input = node.input();

            //Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - no activation additional input
            // - input was optimized
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.is_output() ||
                node.get_dependencies().size() != 1 || input.can_be_optimized())
                return;

            // - check if there is no activation fused already
            // - limit to primitives which implementations support activation fusing
            if (input.get_users().size() != 1 || input.get_fused_activation_func() != activation_none ||
                //TODO: new api needs to be created to read such caps
                //right now use whitelist so no new primitives will be affected in case of lack of fused activation support
                (!input.is_type<batch_norm>() && !input.is_type<concatenation>() && !input.is_type<convolution>() &&
                    !input.is_type<crop>() && !input.is_type<deconvolution>() && !input.is_type<eltwise>() &&
                    !input.is_type<fully_connected>() && !input.is_type<lrn>() && !input.is_type<normalize>() &&
                    !input.is_type<permute>() && !input.is_type<pooling>() && !input.is_type<reorder>() &&
                    !input.is_type<reshape>() && !input.is_type<roi_pooling>() && !input.is_type<scale>() &&
                    !input.is_type<softmax>() && !input.is_type<upsampling>() && !input.is_type<mvn>()))
                return;

            input.set_fused_activation(node.get_primitive()->activation_func, node.get_primitive()->additional_params);
            input.set_output_padding(node.get_output_layout().data_padding);

            p.extract_and_remove(node);
        });
    }

    //This loop tries fusing several reorders one by one (if present) into one reorder
    itr = p.processing_order.begin();
    while (itr != p.processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        program_helpers::do_for_types<reorder>(*node, [&p, is_debug](reorder_node& node)
        {
            auto& input = node.input();

            //Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - input was optimized
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.get_dependencies().size() != 1 ||
                input.can_be_optimized())
                return;

            // - check if previous node is reorder with 1 user
            // - do not fuse if current node has mean subtract
            if (input.get_users().size() != 1 || !input.is_type<reorder>() ||
                node.has_mean() || !node.get_primitive()->subtract_per_feature.empty())
                return;

            input.set_output_layout(node.get_output_layout(), false);
            p.extract_and_remove(node);
        });
    }
    //This loop tries fusing eltwise (sum) with deconvolution
    itr = p.processing_order.begin();
    while (itr != p.processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        fuse_skip_layers(p, node);
    }
}