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

#include "api/CPP/activation.hpp"
#include "api/CPP/eltwise.hpp"
#include "api/CPP/input_layout.hpp"
#include "api/CPP/pooling.hpp"
#include "api/CPP/proposal.hpp"
#include "api/CPP/roi_pooling.hpp"

#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "internal_primitive.h"
#include "internal_primitive_type_base.h"
#include "layout_optimizer.h"
#include "pass_manager.h"
#include "primitive_type.h"
#include "program_dump_graph.h"
#include "program_helpers.h"
#include "program_impl.h"
#include "sliding_window_utils.h"

#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "lstm_inst.h"
#include "lstm_elt_inst.h"
#include "lstm_gemm_inst.h"
#include "mutable_data_inst.h"
#include "primitive_inst.h"
#include "prior_box_inst.h"
#include "reorder_inst.h"
#include "reshape_inst.h"
#include "split_inst.h"
#include "upsampling_inst.h"

#include "gpu/ocl_toolkit.h"

#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>

program_impl::program_impl(engine_impl& engine_ref, topology_impl const& topology, build_options const& options, bool is_internal)
    : engine(&engine_ref), options(options), processing_order(* new nodes_ordering)
{
    set_options();
    prepare_nodes(topology);
    build_program(is_internal);
}

program_impl::program_impl(engine_impl& engine_ref, std::set<std::shared_ptr<program_node>> const& nodes, build_options const& options, bool is_internal)
    : engine(&engine_ref), options(options), processing_order(*new nodes_ordering)
{
    set_options();
    prepare_nodes(nodes);
    build_program(is_internal);
}

program_node& program_impl::get_node(primitive_id const& id)
{
    try
    {
        return *nodes_map.at(id);
    }
    catch (...)
    {
        throw std::runtime_error("Program doesn't contain primtive node: " + id);
    }
}

program_node const& program_impl::get_node(primitive_id const& id) const
{
    try
    {
        return *nodes_map.at(id);
    }
    catch (...)
    {
        throw std::runtime_error("Program doesn't contain primtive node: " + id);
    }
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
bool program_impl::analyze_output_size_handling_need()
{
    bool handling_needed = false;

    // Calculate output size and compare with specified.
    for (const auto& node : processing_order)
    {
        if (node->is_type<convolution>())
        {
            auto& prim_node = node->as<convolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_output_range<swor_mode::all>(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, prim->dilation, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<deconvolution>())
        {
            auto& prim_node = node->as<deconvolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_needed_input_range(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<pooling>())
        {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({ 0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1] }, 1);

            // TODO: Check compatibility of output size calculation (with caffe).
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                prim_node.input().get_output_layout().size,
                prim->size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    return handling_needed;
}

// create new nodes for a program based on the set of nodes
// method created to be used by propagate_constants to build sub program from constant nodes 
void program_impl::prepare_nodes(std::set<std::shared_ptr<program_node>>const &nodes)
{
    for (const auto& itr : nodes)
    {
        if (itr.get()->is_type<data>())
        {
            get_or_create(
                std::make_shared<input_layout>(itr.get()->id(), itr.get()->as<data>().get_primitive()->mem.get_layout())
            );
        }
        else
        {
            get_or_create(itr->desc);
        }
    }
    for (const auto& node : nodes_map)
    {
        auto node_ptr = node.second;
        if (node_ptr == nullptr)
            throw error("NULL pointer in nodes_map.", CLDNN_ERROR);
        //ToDo: avoid O(n^2) run time here (pass map instead of set?)
        bool found = false;
        for (const auto& src_node : nodes)
        {
            if (src_node == nullptr)
                throw error("NULL pointer in nodes_map.", CLDNN_ERROR);
            if (node.first == src_node->get_primitive()->id)
            {
                copy_node_dependencies(node_ptr.get(), src_node.get());
                found = true;
                break;
            }
        }
        if (!found)
        {
            add_node_dependencies(node_ptr.get());
        }
        if (node_ptr->dependencies.size() == 0)
            inputs.push_back(node_ptr.get());
    }
}

// create all nodes from topology primitives, add dependencies among them and create inputs list
void program_impl::prepare_nodes(topology_impl const &topology)
{
    auto const& topo_map = topology.get_primitives();
    for (const auto& prim : topo_map)
    {
        get_or_create(prim.second);
    }
    add_split_outputs();
    for (const auto& node : nodes_map)
    {
        auto node_ptr = node.second.get();
        if (node_ptr == nullptr)
            throw error("NULL pointer in nodes_map.", CLDNN_ERROR);
        add_node_dependencies(node_ptr);
        if (node_ptr->dependencies.size()==0)
        {
            inputs.push_back(node_ptr);
        }
    }
}

// add node's dependecies from its primitive dependencies
void program_impl::add_node_dependencies(program_node* node)
{
    auto deps = node->get_primitive()->dependencies();
    //add pointers to node's dependencies
    for (auto& dep : deps)
    {
        try {
            auto dep_node = nodes_map.at(dep);
            node->dependencies.push_back(dep_node.get());
            dep_node->users.push_back(node);
        }
        catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + dep +
                " that is input to: " + node->get_primitive()->id);
        }
    }
}

/* helper method for program_impl constructor from list of nodes which
   copies src_node dependecies to the destination node dest_node dependencies.
   But only to those which appaer in this program implementation nodes_map */
void program_impl::copy_node_dependencies(program_node* dest_node, program_node* src_node)
{
    if (dest_node->get_primitive()->id != src_node->get_primitive()->id)
    {
        throw std::runtime_error("Node " + src_node->get_primitive()->id +  " and its copy " + dest_node->get_primitive()->id + " do not match.");
    }
    auto src_deps = src_node->get_dependencies();
    //add pointers to node's dependencies
    for (auto& src_dep : src_deps)
    {
        // do not copy dependencies to nodes which does not belong to the new (subgraph) topology
        if (nodes_map.find(src_dep->get_primitive()->id) == nodes_map.end()) continue;

        try {
            auto dest_dep = nodes_map.at(src_dep->get_primitive()->id);
            dest_node->dependencies.push_back(dest_dep.get());
            dest_dep->users.push_back(dest_node);
        }
        catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + src_dep->get_primitive()->id +
                " that is input to: " + src_node->get_primitive()->id);
        }
    }
}

void program_impl::set_options()
{
    static std::atomic<uint32_t> id_gen{ 0 };
    prog_id = ++id_gen;
    assert(prog_id != 0);

    if ((options.get<build_option_type::tuning_config>()->config.mode == tuning_mode::tuning_tune_and_cache) &&
        !engine->configuration().enable_profiling)
    {
        throw std::invalid_argument("Engine must be created with profiling enabled in tune_and_cache mode!");
    }
}

void program_impl::build_program(bool is_internal)
{
    init_graph();
    {
        pre_optimize_graph(is_internal);
    }
    compile_graph();
    {
        post_optimize_graph(is_internal);
    }
    engine->compile_program(*this);
    this->dump_program("13_finished", true);
    cleanup();
}

void program_impl::init_graph()
{
    replace_nodes();
    handle_detection_output();
    handle_lstm();
    set_outputs();
    processing_order.calc_processing_order(*this);

    dump_program("0_init", true);

    calc_prior_boxes(); dump_program("1_calculated_prior_boxes", true);
    mark_constants();
    mark_data_flow();
    dump_program("2_analyzed_graph", true);
}

void program_impl::pre_optimize_graph(bool is_internal)
{
    trim_to_outputs trim_pass; //trim to outputs
    trim_pass.run(*this); // ToDo remove hidden dependencies from trimm pass
    dump_program("3_trimmed", true);

    add_reshape_to_primitives add_reshape_to_primitives_pass; // add reshape to input/parameters for some primitives
    add_reshape_to_primitives_pass.run(*this);

    processing_order.calculate_BFS_processing_order(); // this method makes sense only for OOOQ (out of order execution queue)

    bool output_size_handling_enabled = analyze_output_size_handling_need();
    for (auto& node : processing_order)
    {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>())
            node->get_output_layout();
    }

    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_primitive_fusing prepare_primitive_fusing_pass;
        prepare_primitive_fusing_pass.run(*this);

        layout_optimizer lo(output_size_handling_enabled);
        reorder_inputs reorder_inputs_pass(lo);
        reorder_inputs_pass.run(*this);

        // this code should be moved to post compilation after kernel selector will support handling reorder bias
        pre_optimize_bias pre_optimize_bias_pass(lo);
        pre_optimize_bias_pass.run(*this);
        dump_program("4_reordered_inputs", true);

        // passes regarding conv + eltwise optimizations

        // shrinking eltwise if users are conv 1x1 with stride > 1 optimization
        eltwise_shrinking eltwise_shrinking_pass;
        eltwise_shrinking_pass.run(*this);

        // trying to set stride to 1x1 by shrinking convolutions before eltwise if doable
        eltwise_remove_stride eltwise_remove_stride_pass;
        eltwise_remove_stride_pass.run(*this);

        prepare_conv_eltw_fusing prepare_conv_eltw_fusing_pass;
        prepare_conv_eltw_fusing_pass.run(*this);
    }

    handle_reshape();

    remove_redundant_reorders remove_redundant_reorders_pass;
    remove_redundant_reorders_pass.run(*this);
    dump_program("5_removed_redundant_reorders", true);

    prepare_padding(output_size_handling_enabled);

    prepare_depthwise_sep_opt prepare_depthwise_sep_opt_pass;
    prepare_depthwise_sep_opt_pass.run(*this);

    if (!is_internal)
    {
        propagate_constants propagate_constants_pass;  // ToDo remove hidden dependencies from propagate_constants pass
        propagate_constants_pass.run(*this);
        dump_program("6_propagated_constants", true);
    }
    //try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_buffer_fusing prepare_buffer_fusing_pass;
        prepare_buffer_fusing_pass.run(*this);
    }

    //check if there exists some layout incompatibilities and add an reorder node if required
    add_required_reorders add_required_reorders_pass;
    add_required_reorders_pass.run(*this);

    dump_program("7_pre_optimized", true);
}

void program_impl::compile_graph()
{
    for (auto& node : processing_order)
    {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>())
        {
            node->get_output_layout();
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty()))
                node->selected_impl = node->type()->choose_impl(*engine, *node);
        }
    }

    dump_program("8_compiled", true);
}

void program_impl::post_optimize_graph(bool is_internal)
{
    layout_optimizer lo;
    post_optimize_weights post_optimize_weights_pass(lo);
    post_optimize_weights_pass.run(*this);
    dump_program("9_reordered_weights", true);

    remove_redundant_reorders remove_redundant_reorders_pass;
    remove_redundant_reorders_pass.run(*this);

    dump_program("10_removed_redundant_reorders", true); //TODO: do we need it at this place also?

    if (!is_internal)
    {
        propagate_constants propagate_constants_pass;  // ToDo remove hidden dependencies from propagate_constants pass
        propagate_constants_pass.run(*this);
        dump_program("11_propagated_constants", true);
    }

    prep_opt_depthwise_sep_post prep_opt_depthwise_sep_post_pass;
    prep_opt_depthwise_sep_post_pass.run(*this);
    dump_program("12_prep_opt_depthwise_sep_post_done", true);

    prepare_memory_dependencies();
}

void program_impl::cleanup()
{
    for (auto& node : processing_order)
        if (!node->is_type<internal_primitive>())
            node->get_output_layout();

    //in debug build, at the end, mark all nodes as outputs so user can query for buffers of all not-optimized nodes, including internal ones etc.
    if (is_debug_build())
    {
        for (auto& node : processing_order)
        {
            if (!node->is_output())
            {
                node->set_output(true);
                outputs.push_back(node);
            }
        }
    }
}

std::string get_id_string(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

void program_impl::add_split_outputs() {
    auto itr = nodes_map.begin();
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        if (node->is_type<split>())
        {
            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            //create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++)
            {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];

                //create dummy crop primitive and add it to nodes map
                auto crop_prim = std::make_shared<crop>(output_id, input_id, tensor{ 1,1,1,1 }, split_prim->output_offsets[i]);
                get_or_create(crop_prim);
            }
        }
    }
}

void program_impl::replace_nodes()
{
    auto itr = nodes_map.begin();
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        if (node->is_type<split>())
        {
            //check if split is not used by any primitive, as it will be optimized
            if (node->get_users().size() != 0)
                throw std::logic_error("Split layer cannot be used directly! Please use split output \"" + node->id() + ":<split_output_id>\"!");

            //get_output size and validate split primitive inputs
            auto output_layout = node->get_output_layout();
            auto output_layout_size = output_layout.size;

            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            //create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++)
            {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];

                auto node_ptr = nodes_map.find(output_id)->second;

                //calculate crop reference input size
                tensor reference_input_size;

                // For all the split offsets before the last split offset, the size can be calculated
                // size_of_offset[n] = offset[n + 1] - offset[n];
                if (i != (split_num - 1))
                {
                    reference_input_size += split_prim->output_offsets[i + 1] - split_prim->output_offsets[i];
                }
                // For the last split i.e. size[split_num - 1] = split_input.size - offsets[n];
                else
                {
                    reference_input_size += output_layout_size - split_prim->output_offsets[i];
                }

                // For all the other dimensions, copy from the split_input
                for (int dimension = 0; dimension < CLDNN_TENSOR_DIM_MAX; dimension++)
                {
                    reference_input_size.raw[dimension]
                        = (reference_input_size.raw[dimension] == 0) ? output_layout_size.raw[dimension] : reference_input_size.raw[dimension];
                }

                //update crop primitive and add connections
                node_ptr->set_output_padding(output_layout.data_padding);
                auto crop_prim = node_ptr->as<crop>().typed_desc();
                crop_prim->reference_input = reference_input_size;

                add_connection(node->get_dependency(0), *node_ptr);
            }

            //remove input->split connection and remove original split node
            remove_connection(node->get_dependency(0), *node);
            optimized_out.push_back(node->id());
            nodes_map.erase(node->id());
            continue;
        }

        //find upsampling primitives with bilinear filtering and create deconvolution with proper weights instead
        if (node->is_type<upsampling>())
        {
            auto upsampling_prim = node->as<upsampling>().typed_desc();

            if (upsampling_prim->sample_type != upsampling_sample_type::bilinear)
                continue;

            //check if num_filter is not 0 (required for bilinear upsampling)
            if (upsampling_prim->num_filter == 0)
                throw std::logic_error("num_filter in upsampling cannot be 0 in bilinear filtering mode in \"" + node->id() + "\"!");

            primitive_id upsampling_id = node->id();
            auto& input_node = node->get_dependency(0);

            primitive_id input_id = upsampling_prim->input[0];
            auto num_filter = upsampling_prim->num_filter;

            //setting deconvolution parameters based on upsampling input
            auto scale = static_cast<tensor::value_type>(upsampling_prim->scale);
            tensor stride(1, 1, scale, scale);
            auto offset = static_cast<tensor::value_type>(std::ceil((scale - 1) / 2.f));
            tensor input_offset(0, 0, -offset, -offset);

            //setting weights for deconvolution
            auto kernel_size = static_cast<tensor::value_type>((2 * scale) - (scale % 2));
            layout weights_layout(data_types::f32, format::bfyx, tensor(1, 1, kernel_size, kernel_size));

            std::vector<primitive_id> weights_vec;
            for (uint32_t weights_idx = 0; weights_idx < num_filter; weights_idx++)
            {
                memory_impl::ptr data_to_allocate = engine->allocate_memory(weights_layout);
                mem_lock<float> dst{ data_to_allocate };
                float *dst_data = dst.data();
                //initialize with bilinear weights data
                auto f = static_cast<uint32_t>(std::ceil(kernel_size / 2.0f));
                float c = (2 * f - 1 - f % 2) / (2.f * f);
                float x = 0.f;
                float y = 0.f;
                for (size_t i = 0; i < weights_layout.count(); ++i) {
                    x = static_cast<float>(i % kernel_size);
                    y = static_cast<float>((i / kernel_size) % kernel_size);
                    dst_data[i] = (1 - std::abs(x / f - c)) * (1 - std::abs(y / f - c));
                }

                //create weights primitive, with dummy memory which will be replaced in firther step
                primitive_id weights_id = upsampling_id + "_deconvolution_weights" + std::to_string(weights_idx);
                layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));
                float zero = 0.f;
                auto weights_prim = std::make_shared<data>(weights_id, memory::attach(dummy_layout, &zero, 1));
                get_or_create(weights_prim);

                weights_vec.push_back(weights_id);

                auto weights_node_ptr = nodes_map.find(weights_id)->second;

                //attach weights buffer
                auto& data_node = weights_node_ptr->as<data>();
                data_node.attach_memory(*data_to_allocate, false);
            }

            //remove upsampling node, rename it and move to the optimized list
            remove_connection(node->get_dependency(0), *node);
            auto rename_id = upsampling_id + "_tmp";
            rename(*node, rename_id);

            //create deconvolution primitive
            auto deconv_prim = std::make_shared<deconvolution>(upsampling_id, input_id, weights_vec, stride, input_offset);
            get_or_create(deconv_prim);

            auto deconv_node_ptr = nodes_map.find(upsampling_id)->second;

            auto upsampling_node_ptr = nodes_map.find(rename_id)->second;
            replace_all_usages(*upsampling_node_ptr, *deconv_node_ptr);
            optimized_out.push_back(rename_id);
            nodes_map.erase(rename_id);

            //add connections input->deconvolution and weights->deconvolution
            add_connection(input_node, *deconv_node_ptr);

            for (uint32_t weights_idx = 0; weights_idx < num_filter; weights_idx++)
            {
                auto weights_node_ptr = nodes_map.find(weights_vec[weights_idx])->second;
                add_connection(*weights_node_ptr, *deconv_node_ptr);
            }
            continue;
        }

        //find deconvolution primitives with stride 1 and change them to convolution with trasposed weights
        if (node->is_type<deconvolution>())
        {
            if (!options.get<build_option_type::optimize_data>()->enabled())
                continue;

            auto deconv_prim = node->as<deconvolution>().typed_desc();

            //limit optimization to stride = 1
            if (deconv_prim->stride.spatial[0] != 1 || deconv_prim->stride.spatial[1] != 1 || deconv_prim->gradient())
                continue;

            primitive_id deconv_id = node->id();
            auto& input_node = node->get_dependency(0);

            primitive_id input_id = deconv_prim->input[0];

            //setting convolution parameters based on deconvolution params
            auto stride = deconv_prim->stride;
            auto weights = deconv_prim->weights;
            std::vector<primitive_id> weights_vec;
            for (auto& weights_id : weights)
                weights_vec.push_back(weights_id);
            auto biases = deconv_prim->bias;
            std::vector<primitive_id> bias_vec;
            for (auto& bias_id : biases)
                bias_vec.push_back(bias_id);
            auto input_offset = deconv_prim->input_offset;
            auto with_activation = deconv_prim->with_activation;
            auto activation_negative_slope = deconv_prim->activation_negative_slope;
            auto output_padding = deconv_prim->output_padding;

            //remove deconvolution node and its connections to weights and biases, rename it and move to the optimized list
            tensor filter_size = { 1, 1, 1, 1 };
            remove_connection(node->get_dependency(0), *node);
            for (auto& weights_id : weights_vec)
            {
                auto weights_node_ptr = nodes_map.find(weights_id)->second;
                remove_connection(*weights_node_ptr, *node);
                //get filter spatial sizes for input offset adjustment, perform this only once as all filters shouls have same size
                if (weights_id == weights_vec[0])
                    filter_size = weights_node_ptr->get_output_layout().size;
            }

            input_offset.spatial[0] = std::abs(input_offset.spatial[0]) - (filter_size.spatial[0] - 1);
            input_offset.spatial[1] = std::abs(input_offset.spatial[1]) - (filter_size.spatial[1] - 1);

            if (!bias_vec.empty())
            {
                for (auto& bias_id : bias_vec)
                {
                    auto bias_id_node_ptr = nodes_map.find(bias_id)->second;
                    remove_connection(*bias_id_node_ptr, *node);
                }
            }
            auto rename_id = deconv_id + "_tmp";
            rename(*node, rename_id);

            //create convolution primitive
            if (biases.size() != 0)
            {
                auto conv_prim = std::make_shared<convolution>(deconv_id, input_id, weights_vec, bias_vec,
                    stride, input_offset, tensor{ 1, 1, 1, 1 }, with_activation, activation_negative_slope, output_padding);
                get_or_create(conv_prim);
            }
            else
            {
                auto conv_prim = std::make_shared<convolution>(deconv_id, input_id, weights_vec,
                    stride, input_offset, tensor{ 1, 1, 1, 1 }, with_activation, activation_negative_slope, output_padding);
                get_or_create(conv_prim);
            }

            auto conv_node_ptr = nodes_map.find(deconv_id)->second;
            auto conv_node = &conv_node_ptr->as<convolution>();
            conv_node->set_transposed(true);

            //add connections input->convolution, weights->convolution and bias->convolution
            add_connection(input_node, *conv_node_ptr);

            for (auto& weights_id : weights_vec)
            {
                auto weights_node_ptr = nodes_map.find(weights_id)->second;
                add_connection(*weights_node_ptr, *conv_node_ptr);
            }

            if (!bias_vec.empty())
            {
                for (auto& bias_id : bias_vec)
                {
                    auto bias_id_node_ptr = nodes_map.find(bias_id)->second;
                    add_connection(*bias_id_node_ptr, *conv_node_ptr);
                }
            }

            auto deconv_node_ptr = nodes_map.find(rename_id)->second;
            replace_all_usages(*deconv_node_ptr, *conv_node_ptr);
            optimized_out.push_back(rename_id);
            nodes_map.erase(rename_id);

            continue;
        }
    }
}

void program_impl::handle_detection_output()
{
    auto itr = nodes_map.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = *(*node_itr).second;
        // Create second part detection output primitive and replace nodes names - do it only once
        if ((options.get<build_option_type::detection_output_gpu>()->enabled()) &&
            (node.is_type<detection_output>()) &&
            (node.id().find("_pre") == std::string::npos))    //ToDo: this will fail if user will name the primitive with using _pre like do_pre
                                                              //      we need to use node mark() or some other idea to prevent it   
        {
            // rename detection output
            const primitive_id detect_out_node_name = node.id();
            const primitive_id new_primitive_id = detect_out_node_name + "_pre";
            rename(node, new_primitive_id);

            auto detect_out_prim = node.as<detection_output>().typed_desc();
            // Create new primitive, "keep top k" part of detection output
            // ToDo: add a default parameters to the detection_output_sort class constructor to get rid off this initialization from here
            auto detect_out_sort_prim = std::make_shared<detection_output_sort>(
                detect_out_node_name,
                node.id(),
                // not important params here - it will be set during "primitive_impl* create" func in "detection_output_sort_gpu"
                0,      // num_images
                0,      // num_classes
                0,      // keep_top_k
                false,  // share_location
                0,      // top_k
                -1,     // background_label_id
                detect_out_prim->output_padding);

            get_or_create(detect_out_sort_prim);

            auto sort_node = nodes_map.find(detect_out_node_name)->second;

            // Add connection to second part of detection output
            if (node.get_users().size())
            {
                add_intermediate(*sort_node, *(node.get_users().front()), 0, false);
            }
            else
            {
                add_connection(node, *sort_node);
            }
        }
    }
}

void program_impl::handle_lstm()
{
    bool has_lstm_children;
    auto itr = nodes_map.begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != nodes_map.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;
        has_lstm_children = false;
        // replace lstm node with lstm_gemm and lstm_elt nodes
        if (node->is_type<lstm>()) {
            bool initial_hidden_term = node->as<lstm>().initial_hidden_term();
            bool initial_cell_term = node->as<lstm>().initial_cell_term();
            bool bias_term = node->as<lstm>().bias_term();
            auto lstm_prim = node->as<lstm>().typed_desc();
            primitive_id weights_id = lstm_prim->weights;
            primitive_id recurrent_id = lstm_prim->recurrent;
            primitive_id bias_id = bias_term ? lstm_prim->bias : "";
            primitive_id initial_hidden_id = initial_hidden_term ? lstm_prim->initial_hidden : "";
            primitive_id initial_cell_id = initial_cell_term ? lstm_prim->initial_cell : "";

			//removing connection with weights to get proper dependency order for next operations
            remove_connection(*nodes_map.at(weights_id), *node);
            remove_connection(*nodes_map.at(recurrent_id), *node);
            if (bias_term)
                remove_connection(*nodes_map.at(bias_id), *node);
            if (initial_hidden_term)
                remove_connection(*nodes_map.at(initial_hidden_id), *node);
            if (initial_cell_term)
                remove_connection(*nodes_map.at(initial_cell_id), *node);

            //calculating sizes
            auto input_size = node->get_dependency(0).get_output_layout().size;
            auto recurrent_size = nodes_map.at(recurrent_id)->get_output_layout().size;

            // hidden tensor size = [batch, seq, hidden_size, direction]
            // the output of the element wise operation is cropped and used in the next time step
            // sequence_len = 1 and direction = 1. The backward pass is separated from the forward pass
            auto hidden_size = tensor(input_size.batch[0], 1, recurrent_size.spatial[0], 1);

            size_t directions = recurrent_size.feature[0];
            size_t input_directions = input_size.spatial[1];
            size_t num_input_dependencies = node->get_dependencies().size();
            size_t input_vector_size = node->as<lstm>().sequence_len();
            size_t sequence_len = input_vector_size;

            // Calculate the input sequence length for the lstm node
            // Case 1: If the input comes in as a concatenated input i.e. the
            // input is not divided into sequence elements
            if (input_vector_size == 1 && num_input_dependencies == 1)
            {
	            // Either the input actually has 1 sequence element
	            auto& input = node->get_dependency(0);
	            auto input_layout = input.get_output_layout();
                tensor input_tensor = input_layout.size;
				
	            // Get the sequence length from the input to LSTM
	            sequence_len = input_layout.size.feature[0];  

	            // If the input's feature/sequence length field is > 1, i.e. If
                // the sequence elements are concatenated into one single input
                // then it has to be split into individual sequence elements
	            if (sequence_len > 1)
	            {                
                    for (size_t sequence_element = 0; sequence_element < sequence_len; sequence_element++)
                    {
                        primitive_id crop_id = input.id() + ":crop:" + get_id_string(sequence_element);
                        tensor crop_tensor{ input_tensor.batch[0], 1, input_tensor.spatial[0], input_tensor.spatial[1] };
                        tensor offset_tensor{ 0, static_cast<tensor::value_type>(sequence_element), 0, 0 };
                        auto input_crop = std::make_shared<crop>(crop_id, input.id(), crop_tensor, offset_tensor);
                        auto& input_crop_node = get_or_create(input_crop);

                        // Add the crop nodes as user for input
                        add_connection(node->get_dependency(0), input_crop_node);

                        // Connect crop with lstm
                        add_connection(input_crop_node, *node);
                    }

                    // We have the sequence elements (cropped inputs) as input to LSTM. 
                    // The original input is no longer a dependency to LSTM. 
                    // Remove the input node as a dependency to LSTM
                    remove_connection(node->get_dependency(0), *node);

		            // Update the total no. of input dependecies
		            num_input_dependencies = node->get_dependencies().size();
	            }
            }

            //if the sequence has a single element but it has multiple inputs then
            //the parent of this lstm is an lstm node. If this is a bidirectional lstm
            //then the sequence length is the number of dependencies divided by 2.
            else if (input_vector_size == 1 && num_input_dependencies > 1) 
            {
	            sequence_len = (directions == 1) ? num_input_dependencies : num_input_dependencies / 2;
            }

            //check if this lstm node has an lstm child
            for (auto& user : node->get_users())
            {
                if (user->is_type<lstm>())
                {
                    has_lstm_children = true;
                }
            }

            bool emit_last_cell = lstm_prim->output_selection == cldnn_lstm_output_hidden_cell ||
                                  lstm_prim->output_selection == cldnn_lstm_output_sequence_cell;
            bool emit_sequence = lstm_prim->output_selection == cldnn_lstm_output_sequence_cell ||
                                 lstm_prim->output_selection == cldnn_lstm_output_sequence;

            std::vector<program_node*> cell_list(directions * sequence_len);
            std::vector<program_node*> hidden_list(directions * sequence_len);
            std::map<size_t, std::pair<primitive_id, program_node*>> output_map;
			auto dependencies = node->get_dependencies();

            //lstm expanding
            for (size_t dir = 0; dir < directions; ++dir) {
                auto hidden_id = initial_hidden_id;
                auto cell_id = initial_cell_id;
                for (size_t i = 0; i < sequence_len; ++i) {
                    size_t idx = i + dir * sequence_len;
                    primitive_id lstm_gemm_id = node->id() + ":lstm_gemm" + get_id_string(idx);
                    primitive_id lstm_elt_id = node->id() + ":lstm_elt" + get_id_string(idx);
                    primitive_id crop_id = node->id() + ":crop" + get_id_string(idx);

                    size_t input_idx = i;
                    //for bidirectional lstms, if first LSTM layer then reverse input
                    //for subsequent stacked layers the input is strided on the dir dimension
                    if (directions > 0) {
                        if (num_input_dependencies > sequence_len) { // stacked layer
                            input_idx = dir * sequence_len + i;
                        }
                        else
                        {
                            if ((input_directions < 2) && dir > 0) { // first layer
                                input_idx = sequence_len - i - 1;
                            }
                        }
                    }
                    
		    //primitive_id lstm_gemm_input_id = node->get_dependency(input_idx).get_primitive()->id;
                    //the line below requires an attention: get_org_primitive_id() might not be an actual id of a node (see rename method)
                    //ToDO: ensure that get_org_primitive_id() is suitable here
                    primitive_id lstm_gemm_input_id = node->get_dependency(input_idx).get_org_primitive_id();

                    auto lstm_gemm_node = std::make_shared<lstm_gemm>(lstm_gemm_id, lstm_gemm_input_id, weights_id, recurrent_id, bias_id, hidden_id, (uint32_t)dir);
                    auto &n1 = get_or_create(lstm_gemm_node);

                    auto lstm_elt_node = std::make_shared<lstm_elt>(lstm_elt_id, lstm_gemm_id, cell_id, lstm_prim->clip, lstm_prim->input_forget,
                        lstm_prim->activations, lstm_prim->activation_params, lstm_prim->offset_order, (uint32_t)dir);
                    auto &n2 = get_or_create(lstm_elt_node);
                    //adding lstm_elt as user
                    add_connection(n1, n2);
                    //adding dependecy to lstm_gemm node
                    //input
                    add_connection(node->get_dependency(input_idx), n1);
                    //adding weights and initial values to lstm_gemm
                    add_connection(*nodes_map.at(weights_id), n1);
                    add_connection(*nodes_map.at(recurrent_id), n1);
                    if (bias_term)
                        add_connection(*nodes_map.at(bias_id), n1);

                    //adding cell and hiddens as dependencies
                    if (i > 0)
                    {
                        add_connection(*cell_list[size_t(i - 1) * directions + dir], n2);
                        add_connection(*hidden_list[size_t(i - 1) * directions + dir], n1);
                    }
                    //if initial values are present
                    else
                    {
                        if (initial_hidden_term)
                            add_connection(*nodes_map.at(hidden_id), n1);
                        if (initial_cell_term)
                            add_connection(*nodes_map.at(cell_id), n2);
                    }

                    //lstm_hidden
                    {
                        hidden_id = crop_id + ":hidden";
                        auto crop_hidden = std::make_shared<crop>(hidden_id, lstm_elt_id, hidden_size, tensor{ 0,0,0,0 });
                        auto &n3 = get_or_create(crop_hidden);
                        //adding eltwise as dependency to hidden
                        add_connection(n2, n3);

                        //if parent is lstm adding hiddens as dependency
                        if (has_lstm_children)
                        {
                            for (auto& user : node->get_users())
                            {
                                add_connection(n3, *user);
                            }
                        }
                        hidden_list[i * directions + dir] = &n3;
                        if (i == sequence_len - 1 || emit_sequence)
                        {
                            output_map[i * directions + dir] = {hidden_id, &n3};
                        }
                    }

                    //lstm_cell
                    if (i < sequence_len - 1 || emit_last_cell)
                    {
                        cell_id = crop_id + ":cell";
                        auto crop_cell = std::make_shared<crop>(cell_id, lstm_elt_id, hidden_size, tensor{ 0,1,0,0 });
                        auto &n4 = get_or_create(crop_cell);
                        add_connection(n2, n4);
                        cell_list[i * directions + dir] = &n4;
                        if (i == sequence_len - 1)
                        {
                            output_map[sequence_len * directions + dir] = {cell_id, &n4};
                        }
                    }
                }
            }
            //if there is no next lstm, concatenation is created
            if (!has_lstm_children)
            {
                std::vector<primitive_id> output_ids_offsets;
                for (auto& e : output_map)
                {
                    output_ids_offsets.push_back(e.second.first);
                }
                primitive_id original_id = node->id();
                primitive_id concatenation_id = original_id + ":concat";
                auto concatenation_primitive = std::make_shared<concatenation>(concatenation_id, output_ids_offsets, concatenation::along_f);
                auto &concatenation_node = get_or_create(concatenation_primitive);
                for (auto& e : output_map)
                {
                    add_connection(*e.second.second, concatenation_node);
                }
                if (directions == 2) {
                    // bidirectional support requires concatenations along the direction and sequence axis
                    // instead we can concatenate along the sequence axis and reshape the tensor to the account
                    // for the direction
                    size_t concatenate_len = emit_sequence ? sequence_len : 1;
                    if (emit_last_cell) concatenate_len++;

                    tensor output_size {input_size.batch[0], static_cast<int32_t>(concatenate_len), hidden_size.spatial[0], (int32_t)directions};
                    primitive_id reshape_id = original_id + ":reshape";
                    auto reshape_primitive = std::make_shared<reshape>(reshape_id, concatenation_id, output_size);
                    auto &reshape_node = get_or_create(reshape_primitive);
                    add_connection(concatenation_node, reshape_node);
                    replace_all_usages(*node, reshape_node);
                }
                else
                {
                    replace_all_usages(*node, concatenation_node);
                }
            }
            //removing expanded node
            remove_all_connections(*node);
            nodes_map.erase(node->id());
            continue;
        }
    }

}

void program_impl::set_outputs()
{
    auto outputs_option = options.get<build_option_type::outputs>();
    if (!outputs_option->outputs.empty())
    {
        for (auto const& output : outputs_option->outputs)
        {
            auto o_node = nodes_map.at(output);
            o_node->set_output(true);
            outputs.push_back(o_node.get());
        }
    }
    else
    {
        for (auto& node : nodes_map)
            if (node.second->is_endpoint())
            {
                node.second->set_output(true);
                outputs.push_back(node.second.get());
            }
    }
}

program_impl::nodes_ordering& program_impl::get_processing_order()
{
    return processing_order;
}

const program_impl::nodes_ordering& program_impl::get_processing_order() const
{
    return processing_order;
}

void program_impl::calc_prior_boxes()
{
    auto itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto& node = (*itr++);
        if (!node->is_type<prior_box>())
            continue;

        auto& pb_node = node->as<prior_box>();

        pb_node.calc_result();
        remove_connection(pb_node.input(), pb_node);

        auto& result = pb_node.get_result_buffer();
        result.add_ref(); // need to inc ref count since we will be assigning this memory as cldnn_memory in next line that is not ref_count_obj
        auto cpp_mem = details::memory_c_to_cpp_converter::convert(api_cast(&result));

        auto& data_node = get_or_create(std::make_shared<data>("_cldnn_tmp_" + pb_node.id() + "_result", cpp_mem));
        replace(pb_node, data_node);
    }
}

void program_impl::mark_constants()
{
    for (auto& node : processing_order)
    {
        if (node->dependencies.empty())
            continue;
        if (node->is_type<prior_box>())
            continue;

        node->constant = true;
        for (auto& dep : node->get_dependencies())
        {
            if (!dep->constant)
            {
                node->constant = false;
                break;
            }
        }
    }
}

void program_impl::mark_data_flow()
{
    std::list<program_node*> stack;
    for (auto const& node : processing_order)
    {
        if ((node->is_endpoint() && !node->constant) || node->is_type<mutable_data>())
        {
            stack.push_back(node);
            node->data_flow = true;
            node->mark();
        }
    }

    while (!stack.empty())
    {
        auto node = stack.front();
        stack.pop_front();

        size_t dep_idx = 0;
        size_t inputs_count = (node->is_type<internal_primitive>() ? node->get_dependencies().size() : node->get_primitive()->input.size());
        //TODO: remove this hack after addition of constants propagation pass
        //LK: constant propagation pass exists, so is it safe to remove it?
        if (node->is_type<detection_output>() || node->is_type<proposal>())
            inputs_count = 2; //ignore third input as it is related to prior boxes (i.e. concat of prior-boxes)

        for (auto dep : node->get_dependencies())
        {
            bool data_flow = (dep_idx < inputs_count && !dep->constant);
            ++dep_idx;
            if (!data_flow)
                continue;

            dep->data_flow = data_flow;

            if (dep->is_marked())
                continue;

            stack.push_back(dep);
            dep->mark();
        }
    }

    for (auto& node : processing_order)
    {
        assert(!node->constant || !node->data_flow); //node which is constant cannot be marked as data flow
        node->unmark();
    }
}

void add_memory_dependency(program_node* node, program_node* dep)
{
    if (node->can_be_optimized() ||
        !dep->can_be_optimized())
    {
        node->add_memory_dependency(dep->id());
    }
    else
    {
        if (node->id() == dep->id())
        {
            return;
        }
        for (auto subdep : dep->get_dependencies())
        {
            add_memory_dependency(node, subdep);
            add_memory_dependency(subdep, node);
        }
    }
}

void program_impl::basic_memory_dependencies()
{
    auto itr = processing_order.begin();
    std::vector<primitive_id> past_outputs;
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;

        //data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies())
        {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs, this output
        // has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}

void program_impl::skipped_branch_memory_dependencies()
{
    auto itr = processing_order.begin();
    // Primitive A can't use primitive B buffer if processing_num(B) < processing_num(A) and any of B users processing_num > processing_num(AB)
    // Otherwise it could override data that has to be used in the future.
    // TODO: improve algorithm to to O(n*log(n))
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;
        auto itr2 = processing_order.begin();
        if (itr2 == itr)
            continue;
        while (itr2 != processing_order.end())
        {
            auto& node2 = *itr2;
            itr2++;
            if (processing_order.get_processing_number(node2) < processing_order.get_processing_number(node))
            {
                // if at least one user will be processed after 'node', node2 has to be added to forbiden list
                for (auto usr : node2->get_users())
                {
                    if (processing_order.get_processing_number(usr) > processing_order.get_processing_number(node))
                    {
                        add_memory_dependency(node, node2);
                        add_memory_dependency(node2, node);
                        break;
                    }
                }
            }
        }
    }
}

void program_impl::oooq_memory_dependencies()
{
    auto itr = processing_order.begin();
    // This order let us build dependencies based on syncing points.
    // Set of nodes between two syncing points will be called sync_region.
    // Major rules is: can't share resource with nodes in my sync_region

    int32_t last_barrier = 0;
    bool needs_barrier = false;
    std::vector<cldnn::program_node*> sync_region;
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;

        // if any of dep has proccess num after barrier -> needs barrier
        for (auto dep : node->get_dependencies())
        {
            if (processing_order.get_processing_number(dep) >= last_barrier)
            {
                needs_barrier = true;
                break;
            }
        }

        if (needs_barrier)
        {
            last_barrier = processing_order.get_processing_number(node);
            needs_barrier = false;
            // add each pair bi-direction dependency
            for (auto nd1 = sync_region.begin(); nd1 + 1 != sync_region.end(); nd1++)
            {
                for (auto nd2 = nd1 + 1; nd2 != sync_region.end(); nd2++)
                {
                    add_memory_dependency(*nd1, *nd2);
                    add_memory_dependency(*nd2, *nd1);
                }
            }

            // collect dependencies of every node in sync region
            std::vector<cldnn::program_node*> deps;
            for (auto& nd_in_region : sync_region)
                for (auto& dep : nd_in_region->get_dependencies())
                    deps.emplace_back(dep);


            for (auto& nd_in_region : sync_region)
                for (auto& dep : deps)
                {
                    add_memory_dependency(nd_in_region, dep);
                    add_memory_dependency(dep, nd_in_region);
                }

            sync_region.clear();
        }
        sync_region.push_back(node);
    }
}

void program_impl::prepare_memory_dependencies()
{
    if (!get_engine().configuration().enable_memory_pool)
        return;

    basic_memory_dependencies();
    skipped_branch_memory_dependencies();
    oooq_memory_dependencies();
}

std::string program_impl::get_memory_dependencies_string() const
{
    std::string mem_dep = "Memory dependencies/restrictions:\n";
    auto itr = processing_order.begin();
    while (itr != processing_order.end())
    {
        auto& node = *itr;
        itr++;
        mem_dep = mem_dep.append("primitive: ").append(node->id()).append(" restricted list: ");
        for (auto it : node->get_memory_dependencies())
            mem_dep == mem_dep.append(it).append(", ");
        mem_dep = mem_dep.append("\n");
    }
    return mem_dep;
}

void program_impl::handle_reshape()
{
    //reshape primitive by definition does not change underlying data, only shape description
    //however during graph initialization and data optimization the layouts can be changed without user's knowledge,
    //when reshape is followed by reorder, it is likely that reorder's output will not be as expected (for example reshape with flattened shape)
    //this pass resolved the issue by changing graph in the following way
    //- in case reshape has multiple users with reshape->reorder sequence, it will be splitted to multiple reshape primitives with single user
    //- in case of reshape->reorder sequence, the additional reorder before reshape will be added,
    //  if last reorder does not contain padding or mean subtract, it will be removed later in the graph

    for (const auto& node : processing_order)
    {
        if (node->is_type<reshape>())
        {
            auto& input_node = node->get_dependency(0);

            if (input_node.is_type<reorder>())
                continue;

            node->get_output_layout();
            if (node->as<reshape>().is_in_place())
                node->optimized = true;

            //vector for storing nodes that are reorder type, for which splitted primitives are needed (except for the first one where orginal reshape will be used)
            std::vector<program_node*> reorder_node_to_split;

            //find the users of reshape that are reorder type, if none present then skip the current node
            for (const auto& user : node->get_users())
            {
                if (user->is_type<reorder>())
                    reorder_node_to_split.push_back(user);
            }

            if (!reorder_node_to_split.empty())
            {
                auto& prim_node = node->as<reshape>();
                const auto& prim = prim_node.get_primitive();
                auto output_shape = prim->output_shape;

                //vector for storing reshape nodes to connect to new reorder nodes (if needed)
                std::vector<program_node*> reorder_reshape_nodes;

                bool skip_first_user = false;
                auto reshape_users = node->get_users();
                for (const auto& user : reshape_users)
                {
                    //reshape node for first user will be the orginal reshape from the graph
                    if (!skip_first_user)
                    {
                        if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) != reorder_node_to_split.end())
                            reorder_reshape_nodes.push_back(node);
                        skip_first_user = true;
                        continue;
                    }

                    //other reshapes will be clones of the orginal one connected to reshape->reorder sequences
                    if (std::find(reorder_node_to_split.begin(), reorder_node_to_split.end(), user) != reorder_node_to_split.end())
                    {
                        auto new_reshape = std::make_shared<reshape>("_reshape_split_" + user->id() + "_" + node->id(), input_node.id(), output_shape);
                        auto& new_reshape_node = get_or_create(new_reshape);
                        add_connection(input_node, new_reshape_node);
                        user->replace_dependency(0, new_reshape_node);
                        processing_order.insert(std::next(processing_order.get_processing_iterator(input_node)), &new_reshape_node);
                        reorder_reshape_nodes.push_back(&new_reshape_node);
                    }
                }

                //add new reorder nodes to proper reshape node
                auto reshape_reorder_id = 0;
                for (const auto& reorder_node : reorder_node_to_split)
                {
                    /*
                    auto new_reshape = std::make_shared<reshape>("_reshape_split_" + user->id() + "_" + node->id(), input_node.id(), output_shape);
                    auto& new_reshape_node = get_or_create(new_reshape);
                    add_connection(input_node, new_reshape_node);
                    user->replace_dependency(0, new_reshape_node);
                    processing_order.insert(std::next(processing_order.get_processing_iterator(input_node)), &new_reshape_node);
                    reorder_reshape_nodes.push_back(&new_reshape_node);
                    */
                    auto& reorder_reshape_node = reorder_reshape_nodes[reshape_reorder_id];
                    auto reshape_in_layout = reorder_node->get_output_layout();
                    auto reshape_input = std::make_shared<reorder>("_reshape_input_" + reorder_node->id() + "_" + reorder_reshape_node->id(), input_node.id(),
                        reshape_in_layout.format, reshape_in_layout.data_type);
                    auto& reshape_input_node = get_or_create(reshape_input);
                    add_intermediate(reshape_input_node, *reorder_reshape_node, 0, reshape_input_node.dependencies.empty());
                    reshape_reorder_id++;
                }
            }

            auto reshape_layout = node->get_output_layout();
            if (!(node->is_output()) && (reshape_layout.format != cldnn::format::bfyx))
            {
                auto bfyx_layout = layout({ reshape_layout.data_type, cldnn::format::bfyx, reshape_layout.size });
                //when some primitive does an implicit reorder to some other format then we lose the info about pitches in reshape stage
                //we assume user provides the input vector in bfyx
                if (!program_helpers::are_layouts_identical(reshape_layout, bfyx_layout).second)
                {
                    auto reshape_input = std::make_shared<reorder>("_reshape_input_" + node->id(), input_node.id(), cldnn::format::bfyx, reshape_layout.data_type);
                    auto& reshape_input_node = get_or_create(reshape_input);
                    add_intermediate(reshape_input_node, *node, 0, reshape_input_node.dependencies.empty());

                    auto reshape_users = node->get_users();
                    for (const auto& user : reshape_users)
                    {
                        size_t idx = 0;
                        for (size_t i = 0; i < user->get_dependencies().size(); i++)
                        {
                            auto& input = user->get_dependency(i);
                            if (input.id() == node->id()) {
                                idx = i;
                                break;
                            }
                        }
                        auto reshape_output = std::make_shared<reorder>("_reshape_output_" + node->id(), user->id(), reshape_layout.format, reshape_layout.data_type);
                        auto& reshape_output_node = get_or_create(reshape_output);
                        add_intermediate(reshape_output_node, *user, idx, reshape_output_node.dependencies.empty());
                    }
                }
            }
        }
    }
}

void program_impl::apply_needed_padding(program_node& node, program_node& prev_node,
    const padding& needed_padding)
{
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>() || prev_node.is_type<mutable_data>())
    {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_input_" + node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program_impl::prepare_padding(bool output_size_handling_enabled)
{
    if (output_size_handling_enabled)
    {
        // Prepare upper padding for primitives that support output_size parameter.
        for (const auto& node : processing_order)
        {
            if (node->is_type<convolution>())
            {
                auto& prim_node = node->as<convolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(
                    prim_node.input().get_output_layout(),
                    prim->output_size, filter_size, prim->input_offset, prim->stride, prim->dilation, false, 1);
                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
            else if (node->is_type<deconvolution>())
            {
                auto& prim_node = node->as<deconvolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(
                    prim_node.input().get_output_layout(),
                    prim->output_size, filter_size, prim->input_offset, prim->stride, { 1, 1, 1, 1 }, true, 1);

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
            else if (node->is_type<pooling>())
            {
                auto& prim_node = node->as<pooling>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                // NOTE: Currently there is no pooling implementation/pooling mode which does not check input data range.
                // There is no need to add padding requirements on pooling inputs.
                //auto needed_padding = calc_sliding_window_needed_input_padding(
                //    prim_node.input().get_output_layout(),
                //    prim->output_size, prim->size, prim->input_offset, prim->stride, {1, 1, 1, 1}, false, 1);
                auto needed_padding = prim_node.input().get_output_layout().data_padding;

                apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            }
        }
    }

    // Prepare optimized padding for bfyx convolution.
    for (auto& pair : nodes_map)
    {
        if (pair.second->type() != convolution::type_id())
            continue;

        auto& node = pair.second->as<convolution>();
        if (node.get_dependencies().empty())
            continue;

        auto conv = node.get_primitive();
        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx
            && conv_layout.format != cldnn::format::bf8_xy16
            && conv_layout.format != cldnn::format::byxf_af32
            && conv_layout.format != cldnn::format::fs_bs_yx_bsv4_fsv32
            && conv_layout.format != cldnn::format::b_fs_yx_fsv4)
        {
            continue;
        }

        // We shoudn't apply any padding to nodes which are marked as outputs
        if (conv_input_node.is_output())
            continue;

        // Calculating input padding needed for convolution
        auto& filter_node = node.as<convolution>().weights(0);
        auto filter_prim = filter_node.get_primitive();

        layout filter_layout = filter_node.get_output_layout();

        // convolution have only one input primitive
        auto prev_prim_output_layout = conv_input_node.get_output_layout();

        // Compute initial required paddings for primitive used as input for convolution.
        auto input_offset = conv->input_offset;
        auto stride = conv->stride;
        auto dilation = conv->dilation;

        auto input_limit_x = input_offset.spatial[0] + (conv_layout.size.spatial[0] - 1) * stride.spatial[0] + (filter_layout.size.spatial[0] - 1) * dilation.spatial[0] + 1;
        auto input_limit_y = input_offset.spatial[1] + (conv_layout.size.spatial[1] - 1) * stride.spatial[1] + (filter_layout.size.spatial[1] - 1) * dilation.spatial[1] + 1;

        auto left_padding = std::max(-input_offset.spatial[0], 0);
        auto top_padding = std::max(-input_offset.spatial[1], 0);
        auto right_padding = std::max(input_limit_x - prev_prim_output_layout.size.spatial[0], 0);
        auto bottom_padding = std::max(input_limit_y - prev_prim_output_layout.size.spatial[1], 0);

        // Adjust right padding, so entire buffer size in X dimension is properly aligned.
        // TODO: NOTE: Will be reenabled with next check-in once heuristic for line-aligned algorithm will be added.
        //auto needed_buffer_size_x = static_cast<cldnn::tensor::value_type>(
        //    round_up_to(left_padding + prev_prim_output_layout.size.spatial[0] + right_padding, 16));
        //right_padding = needed_buffer_size_x - left_padding - prev_prim_output_layout.size.spatial[0];

        cldnn::padding needed_padding({ 0, 0, left_padding, top_padding }, { 0, 0, right_padding, bottom_padding }, 0);
        needed_padding = padding::max(prev_prim_output_layout.data_padding, needed_padding);

        apply_needed_padding(node, conv_input_node, needed_padding);
    }
}

void program_impl::reverse_connection(program_node& dep_node, program_node& user_node)
{
    if (std::find(dep_node.users.begin(), dep_node.users.end(), &user_node) != dep_node.users.end())
    {
        remove_connection(dep_node, user_node);
        add_connection(user_node, dep_node);
    }
    else
        throw std::runtime_error("Trying to reverse connection, but nodes are wrongly or not connected.");
}

program_node& program_impl::get_or_create(std::shared_ptr<primitive> prim)
{
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = prim->type->create_node(*this, prim);
    nodes_map.insert(itr, { prim->id, new_node });
    return *new_node;
}

void program_impl::add_intermediate(program_node& node, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep)
{
    if (connect_int_node_with_old_dep && !node.dependencies.empty())
        throw std::invalid_argument("Node which is about to be added inbetween two other nodes should not have any existing dependencies");

    auto& prev = next.get_dependency(prev_idx);
    //firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    if (connect_int_node_with_old_dep)
    {
        add_connection(prev, node);
/*
        // LK: I assume here that the node which is added does not exist yet, is it true?
        auto tmp = processing_order.get_processing_iterator(node);
        if (tmp != processing_order.end())
            processing_order.erase(tmp);
*/
        if (processing_order.size() != 0)
        {
            auto itr = processing_order.get_processing_iterator(prev);
            processing_order.insert(std::next(itr), &node);
        }
    }

    next.replace_dependency(prev_idx, node);
    node.constant = prev.constant;
    node.data_flow = prev.data_flow;
}

void program_impl::add_intermediate(std::shared_ptr<primitive> prim, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep)
{
    add_intermediate(get_or_create(prim), next, prev_idx, connect_int_node_with_old_dep);
}

void program_impl::add_connection(program_node& prev, program_node& next)
{
    prev.users.push_back(&next);
    next.dependencies.push_back(&prev);
}

void program_impl::remove_connection(program_node& prev, program_node& next)
{
    prev.users.remove(&next);
    next.dependencies.erase(std::remove(next.dependencies.begin(), next.dependencies.end(), &prev), next.dependencies.end());
}

void program_impl::remove_all_connections(program_node& node) {
    // since the graph is not topological sorted, we need to remove the node from both dependencies and users
    for (auto &e : node.users) {
        e->dependencies.erase(std::remove(e->dependencies.begin(), e->dependencies.end(), &node), e->dependencies.end());
    }
    for (auto &e : node.dependencies) {
        e->users.remove(&node);
    }
    node.dependencies.clear();
    node.users.clear();
}

void program_impl::rename(program_node & node, primitive_id const & new_id)
{
    if (nodes_map.count(new_id))
        throw std::runtime_error("Trying to rename program_node but node with id " + new_id + " already exists");
    if (node.is_output())
        throw std::invalid_argument("Trying to rename an output node. If you intend to do that, please clear 'output' flag manually.");

    auto node_ptr = nodes_map.find(node.id())->second;
    nodes_map.emplace(new_id, node_ptr);
    nodes_map.erase(node.id());

    if (!node.is_type<internal_primitive>())
        const_cast<primitive_id&>(node.desc->id) = new_id;
    else
        reinterpret_cast<details::internal_program_node_base&>(node).internal_id = new_id;
}

void program_impl::swap_names(program_node& node1, program_node& node2)
{
    const auto _extract_id = [](program_node& node) -> primitive_id&
    {
        if (!node.is_type<internal_primitive>())
            return const_cast<primitive_id&>(node.desc->id);
        else
            return reinterpret_cast<details::internal_program_node_base&>(node).internal_id;
    };

    nodes_map.at(node1.id()).swap(nodes_map.at(node2.id()));
    std::swap(_extract_id(node1), _extract_id(node2));
}

void program_impl::replace_all_usages(program_node & old_node, program_node & new_node)
{
    auto itr = old_node.users.begin();
    bool end = (itr == old_node.users.end());
    while (!end)
    {
        auto& usage = (*itr++);
        end = (itr == old_node.users.end());
        usage->replace_dependency(old_node, new_node);
    }
}

void program_impl::replace(program_node& old_node, program_node& new_node)
{
    if (!new_node.dependencies.empty() || !new_node.users.empty())
        throw std::invalid_argument("Node which is about to replace other node should be detached");

    if (new_node.is_output())
        throw std::invalid_argument("Replacement node shouldn't be marked as an output since it's impossible to rename such node.");

    auto id = old_node.id();
    new_node.output_layout = old_node.get_output_layout();
    new_node.valid_output_layout = old_node.valid_output_layout;

    
    //copy old's dependencies
    while (!old_node.dependencies.empty())
    {
        auto& dep = old_node.dependencies.front();
        add_connection(*dep, new_node);
        remove_connection(*dep, old_node);
    }

    //append users
    for (auto& user : old_node.users)
    {
        new_node.users.push_back(user);
        for (auto& users_dep : user->dependencies)
        {
            if (users_dep == &old_node)
            {
                users_dep = &new_node;
                break;
            }
        }
    }

    old_node.users.clear();

    bool old_was_output = false;
    //copy node's state
    if (old_node.is_output())
    {
        old_was_output = true;
        old_node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &old_node), outputs.end());
    }
    if (new_node.is_input())
        inputs.push_back(&new_node);
    if (old_node.is_input())
        inputs.remove(&old_node);

    new_node.constant = old_node.constant;
    new_node.user_mark = old_node.user_mark;

    processing_order.insert(processing_order.get_processing_iterator(old_node), &new_node);
    if (processing_order.get_processing_iterator(old_node) != processing_order.end())
        processing_order.erase(processing_order.get_processing_iterator(old_node));
    nodes_map.erase(id);
    rename(new_node, id);

    //mark new node as an output after renaming
    if (old_was_output)
    {
        new_node.set_output(true);
        outputs.push_back(&new_node);
    }
}

bool program_impl::remove_if_dangling(program_node& node)
{
    if (!node.users.empty())
        return false;
    if (!node.dependencies.empty())
        return false;

    if (!node.is_output() || is_debug_build())
    {
        if (node.is_input())
            inputs.remove(&node);

        if (std::find(processing_order.begin(), processing_order.end(), &node) != processing_order.end())
            processing_order.erase(processing_order.get_processing_iterator(node));
        optimized_out.push_back(node.id());
        nodes_map.erase(node.id());
    }
    return true;
}

bool program_impl::extract_and_remove(program_node& node)
{
    if (node.get_dependencies().size() != 1)
        return false;

    if (node.is_output() && node.get_dependency(0).is_output() && !is_debug_build()) //TODO: add a mechanism to support removal of nodes which are marked as outputs
        return false;

    if (node.is_output() && !is_debug_build())
    {
        auto& prev = node.get_dependency(0);
        auto node_id = node.id();

        node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &node), outputs.end());

        rename(node, "_cldnn_tmp_" + node_id);
        rename(prev, node_id);

        prev.set_output(true);
        outputs.push_back(&prev);
    }

    auto& input = node.get_dependency(0);
    node.dependencies.clear();
    input.users.remove(&node);

    if (!node.is_endpoint())
        replace_all_usages(node, input);
    else
        remove_if_dangling(node);

    return true;
}

void program_impl::remove_nodes(std::list<program_node*>& to_remove)
{
    for (auto const& node : to_remove)
    {
        if (node->is_input())
            get_inputs().remove(node);
        else
        {
            for (auto& dep : node->dependencies)
                dep->users.remove(node);
        }
        for (auto& user : node->users)
        {
            user->dependencies.erase(std::remove(user->dependencies.begin(),
                user->dependencies.end(), node),
                user->dependencies.end());
        }
        get_processing_order().erase(get_processing_order().get_processing_iterator(*node));
        optimized_out.push_back(node->id());
        nodes_map.erase(node->id());
    }
}

void program_impl::dump_memory_pool() const
{
    if (!get_engine().configuration().enable_memory_pool)
        return;
    auto path = get_dir_path(options);
    if (path.empty())
    {
        return;
    }

    path += "cldnn_memory_pool.log";
    auto dep = get_memory_dependencies_string();
    get_engine().dump_memory_pool(*this, path, dep);
    dump_program("14_memory_pool", true);
}

//TODO: break this function into number of smaller ones + add per-primitive fields (possibly use primitive_inst::to_string?)
void program_impl::dump_program(const char* stage, bool with_full_info, std::function<bool(program_node const&)> const& filter) const
{
    std::string path = get_dir_path(options);
    if (path.empty())
    {
        return;
    }

    std::ofstream graph(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".graph");
    dump_graph_init(graph, *this, filter);

    if (!with_full_info)
    {
        return;
    }

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".info");
    dump_graph_info(graph, *this, filter);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".order");
    dump_graph_processing_order(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".optimized");
    dump_graph_optimized(graph, *this);
}


