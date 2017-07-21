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

#include "program_impl.h"
#include "primitive_inst.h"
#include "layout_optimizer.h"

#include "primitive_type.h"
#include "api/CPP/activation.hpp"
#include "api/CPP/convolution.hpp"
#include "api/CPP/deconvolution.hpp"
#include "api/CPP/data.hpp"
#include "api/CPP/eltwise.hpp"
#include "api/CPP/input_layout.hpp"
#include "api/CPP/pooling.hpp"
#include "api/CPP/reorder.hpp"

#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "crop_inst.h"

#include "kernel_selector_helper.h"
#include "sliding_window_utils.h"

namespace cldnn
{

namespace {

    //helper function for selecting function basing on the type of the given primitive
    //this is the termination case for parameter pack recurrence, see overload below for logic
    template <class... T>
    void do_for_types(program_node&)
    {
        return;
    }

    //helper function for selecting function basing on the type of the given primitive
    //this function should be explicitly given set of types and implicitly set of functions.
    //both sets should have equal size. First function will be called if type of the given primitive
    //will match first explicitly given type, second will be called if it matches second explicitly given
    //type etc.
    //Functions given as arguments should themselves take std::shared_ptr<const T> as argument
    //where T is the type that should be match if this function should be called
    //
    //example:
    // do_for_types<
    //      convolution,
    //      pooling
    //  >(primitive,
    //      [](typed_program_node<convolution>&){ do something if 'primitive' is a convolution },
    //      [](typed_program_node<pooling>&)    { do something if 'primitive' is a pooling }
    //  );
    template <class T, class... RestOfT, class Func, class... RestOfFuncs>
    decltype(static_cast<void>(std::declval<Func>()(std::declval<typed_program_node<T>&>()))) do_for_types(
        program_node& node,
        Func const& func,
        RestOfFuncs const&... rest)
    {
        if (node.type() == T::type_id())
            func(node.as<T>());
        else
            do_for_types<RestOfT...>(node, rest...);
    }

    template <class T>
    struct single_element_container
    {
        single_element_container(T& t) : elem(&t)
        {}

        constexpr size_t size() const { return 1; }
        auto begin() const { return single_element_container(elem); }
        auto end() const { return single_element_container(nullptr); }
        auto& operator ++() { elem = nullptr; return *this; }
        bool operator !=(single_element_container const& sec) { return elem != sec.elem; }

        decltype(auto) operator *() { return *elem; }

    private:
        single_element_container(T* t) : elem(t)
        {}

        T* elem;
    };

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    //It should be used in generic code when there's a need to force vector usage
    //in foreach loop over variable which can in one context be a vector or a scalar
    //in another.
    //example:
    // T t;
    // for (auto& string : wrap_if_single(t.dump()))
    //depending on type T, t.dump() may return either std::string or std::vector<std::string>,
    //to ensure compatibility between these cases, wrap_if_single will create single-element
    //container in case t.dump() would return plain std::string.
    //
    // T& case -> returns container which holds T&
    template <class T>
    auto wrap_if_single(T& t)
    {
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T const& case -> returns container which holds T const&
    template <class T>
    auto wrap_if_single(T const& t)
    {
        return single_element_container<T const>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // T&& case -> returns container which holds new instance of T created by moving given param
    template <class T>
    auto wrap_if_single(T&& t)
    {
        static_assert(meta::always_false_v<T>, "Wrapping temporary object into single_element_container is an error (requires valid reference)");
        return single_element_container<T>(t);
    }

    //helper function which creates single-element array if it's given anything
    //other than std::vector.
    // std::vector case -> does not wrap, returns t as-is
    decltype(auto) wrap_if_single(primitive::fixed_size_vector_ref const& t)
    {
        return t;
    }
}


void program_node::replace_dependency(size_t idx, program_node& new_dep)
{
    if (idx >= dependencies.size())
        return;
    if (dependencies[idx] == &new_dep)
        return;

    dependencies[idx]->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx]);

    dependencies[idx] = &new_dep;
    desc->dependecies()[idx].get() = new_dep.id();
    new_dep.users.push_back(this);
}

void program_node::replace_dependency(program_node const& old_dep, program_node& new_dep)
{
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i] == &old_dep)
            return replace_dependency(i, new_dep);
}

layout program_node::get_output_layout()
{
    if (valid_output_layout)
        return output_layout;

    for (auto dep : dependencies)
        dep->get_output_layout();

    auto new_layout = desc->type->calc_output_layout(*this);
    //TODO: after merging padding into layout, calc_output_layout can now return padding as well
    // for now just ignore it and preserve already set padding value - in future we should probably take care of this
    // situation however.
    new_layout.data_padding = output_layout.data_padding;
    if (new_layout != output_layout) //output_layout has changed! invalidate users
        invalidate_users();

    output_layout = new_layout;
    valid_output_layout = true;
    return std::move(new_layout);
}

program_impl::program_impl(engine_impl::ptr engine, topology_impl const& topology, build_options const& options)
    : engine(engine), options(options), output_size_handling_enabled(true)
{
    init_graph(topology);
    pre_optimize_graph();
    compile_graph();
    post_optimize_graph();
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
void program_impl::analyze_output_size_handling_need()
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

            tensor specified_output_range({0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1]}, 1);

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

            tensor specified_output_range({0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1]}, 1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_needed_input_range(
                prim_node.input().get_output_layout().size,
                filter_size, prim->input_offset, prim->stride, {1, 1, 1, 1}, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
        else if (node->is_type<pooling>())
        {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range({0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1]}, 1);

            // TODO: Check compatibility of output size calculation (with caffe).
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                prim_node.input().get_output_layout().size,
                prim->size, prim->input_offset, prim->stride, {1, 1, 1, 1}, true, 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    output_size_handling_enabled = handling_needed;
}

std::list<std::shared_ptr<program_node>> program_impl::get_nodes() const
{
    std::list<std::shared_ptr<program_node>> ret;

    forward_bfs([&ret, this](program_node& node) {
        ret.push_back(nodes_map.at(node.id()));
    });

    return ret;
}

void program_impl::init_graph(topology_impl const& topology)
{
    auto const& topo_map = topology.get_primitives();
    for (auto const& prim : topo_map)
    {
        auto& n = get_or_create(prim.second);
        inputs.push_back(&n);
    }

    for (auto itr = inputs.begin(); itr != inputs.end(); )
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);
        auto deps = node->get_primitive()->dependecies();
        if (deps.empty())
            continue;

        //add pointers to node's dependencies
        for (auto& dep : deps)
        {
            try {
                auto dep_node = nodes_map.at(dep);
                node->dependencies.push_back(dep_node.get());
                dep_node->users.push_back(node);
            }
            catch(...) {
                throw std::runtime_error("Program doesn't contain primitive: " + dep +
                    " that is input to: " + node->get_primitive()->id);
            }
        }

        //primitive has dependencies so remove it from 'inputs'
        inputs.erase(node_itr);
    }

    set_outputs();
    calc_processing_order();
}

void program_impl::pre_optimize_graph()
{
    trim_to_outputs();

    analyze_output_size_handling_need();

    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        layout_optimizer lo(engine, true, output_size_handling_enabled);
        reorder_inputs(lo);
        // this code should move to post compilation after kernel selector will support handling reorder bias
        pre_optimize_bias(lo);
    }

    prepare_padding();

    //try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        prepare_buffer_fusing();
    }
}

void program_impl::post_optimize_graph()
{
    layout_optimizer lo(engine);
    post_optimize_weights(lo);
    //prepare_padding(); - TODO: padding should be prepare according to the kernels needs
}

void program_impl::compile_graph()
{
    for (auto& node : processing_order)
    {
        node->get_output_layout();
        node->selected_impl = node->type()->choose_impl(*engine, *node);
    }
}

void program_impl::set_outputs()
{
    auto outputs_option = options.get<build_option_type::outputs>();

    // in debug mode select all primitives as output
    if (options.get<build_option_type::debug>()->enabled())
    {
        for (auto& node : nodes_map)
        {
            //do not add cldnn::data as output
            if (node.second->type() == data::type_id())
                continue;

            node.second->set_output(true);
            outputs.push_back(node.second.get());
        }

        return;
    }

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

void program_impl::trim_to_outputs()
{
    backward_bfs(nullptr, [this](program_node& node) {
        if (!node.is_marked())
        {
            processing_order.erase(node.processing_itr);
            optimized_out.push_back(node.id());

            if (node.is_input())
            {
                inputs.remove(&node);
            }
        }
    });

    for (auto const& opt : optimized_out)
    {
        nodes_map.erase(opt);
    }
}

void program_impl::reorder_inputs(layout_optimizer& lo)
{
    //first pass to set layout optimization_attributes for topology
    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == cldnn::convolution::type_id())
        {
            if (prim.as<convolution>().get_primitive()->split() > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::splitted_convolution, 1);
        }
    }

    const auto reorder_input = [this, &lo](typed_program_node<convolution>& conv_node)
    {
        auto conv_prim = conv_node.get_primitive();
        auto& input_node = conv_node.get_dependency(0);

        std::shared_ptr<reorder> new_input = nullptr;

        if (input_node.type() == data::type_id())
        {
            new_input = lo.add_weights_for_optimization(input_node.as<data>().typed_desc(),
                layout_optimizer::data_type::input,
                conv_prim).first;
        }
        else if (input_node.type() == input_layout::type_id())
        {
            new_input = lo.get_reorder(
                input_node.as<input_layout>().get_primitive()->layout,
                input_node.id(),
                layout_optimizer::data_type::input,
                conv_prim).first;
        }
        else if (input_node.type() == reorder::type_id()) //convolution's input is a reorder
        {
            auto reorder_prim = input_node.as<reorder>().typed_desc();
            auto& reorder_input = input_node.get_dependency(0);
            auto reorder_layout = reorder_input.get_output_layout();
            reorder_layout.data_type = reorder_prim->output_data_type;
            new_input = lo.get_reorder(
                reorder_layout,
                reorder_prim->id,
                layout_optimizer::data_type::input,
                conv_prim).first;

            if (new_input) //output format is not optimal
            {
                auto reorder_input_layout = reorder_input.get_output_layout();

                auto opt_layout = layout(new_input->output_data_type, new_input->output_format, reorder_input_layout.size);
                if (reorder_input_layout == opt_layout) //reorder 'breaks' optimal format
                {
                    if (reorder_prim->subtract_per_feature.empty() &&
                        reorder_prim->mean.empty() &&
                        !reorder_prim->output_padding) //just plain reorder
                    {
                        conv_node.replace_dependency(0, reorder_input);
                        new_input = nullptr;
                    }
                    else //change reorder's output layout
                    {
                        reorder_prim->output_format = opt_layout.format;
                        reorder_prim->output_data_type = opt_layout.data_type;
                        new_input = nullptr;
                    }
                }
                else //current reorder gives bad output, simply change it
                {
                    reorder_prim->output_format = opt_layout.format;
                    reorder_prim->output_data_type = opt_layout.data_type;
                    new_input = nullptr;
                }
            }

            input_node.recalc_output_layout();
        }

        if (new_input)
        {
            add_intermediate(new_input, conv_node, 0);
            conv_node.recalc_output_layout();
        }
    };

    const auto reorder_input_detection_output = [this, &lo](typed_program_node<detection_output>& detection_output_node)
    {
        auto detection_output_prim = detection_output_node.get_primitive();
         
        for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++)
        {
            auto& input = detection_output_node.get_dependency(i);
            std::shared_ptr<reorder> new_input = lo.get_reorder(
                input.get_output_layout(),
                input.id(),
                layout_optimizer::data_type::input,
                detection_output_prim).first;

            if (new_input)
            {
                add_intermediate(new_input, detection_output_node, i);
            }
        }
    };

    for (auto& prim : processing_order)
    {
        //there's an assumption that only convolution will take data/input_layout as input
        //exception to that rule would be a convolution which takes a reorder as input - see reoder_input above
        do_for_types<convolution, detection_output>(*prim,
            reorder_input,                  //case for convolution
            reorder_input_detection_output  //case for detection-output
            );
    }
}

void program_impl::pre_optimize_bias(layout_optimizer& lo)
{
    std::list<program_node*> outputs_to_recalc;

    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_bias = [this, &lo, &outputs_to_recalc](program_node& bias, auto& node, layout const& output_layout, size_t dep_idx)
    {
        const auto bias_type = layout_optimizer::data_type::bias;
        if (bias.type() == data::type_id())
        {
            lo.add_weights_for_optimization(
                bias.as<data>().typed_desc(),
                bias_type,
                node.get_primitive(),
                output_layout);
            outputs_to_recalc.push_back(&bias);
        }
        else if (bias.type() == input_layout::type_id())
        {
            auto reorder = lo.get_reorder(
                bias.as<input_layout>().typed_desc()->layout,
                bias.id(),
                bias_type,
                node.get_primitive(),
                output_layout);

            if (reorder.first)
                this->add_intermediate(reorder.first, node, dep_idx);
        }
        else
            throw std::logic_error("Optimization of weights which are neither of type cldnn::data nor cldnn::input_layout!");
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights and biases used by given primitive.
    //argument should match few requirements:
    // - it should be of a form 'typed_program_node<T>&'
    // - both 'T.weights' and 'T.bias' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [this, &add_bias, &outputs_to_recalc](auto& node) -> void
    {
        auto output_layout = node.get_output_layout();

        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
        for (auto i = weights_offset; i < bias_offset; ++i)
        {
            outputs_to_recalc.push_back(&node.get_dependency(i));
        }
        for (auto i = bias_offset; i < node.get_dependencies().size(); ++i)
        {
            add_bias(node.get_dependency(i), node, output_layout, i);
        }
    };

    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            prep_opt(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            prep_opt(prim.as<deconvolution>());
        }
        else if (prim.type() == fully_connected::type_id())
        {
            prep_opt(prim.as<fully_connected>());
        }
    }

    //all optimizing primitives has been added and inputs for all primitives has been updated.
    //run reorders and replace cldnn::data::mem
    lo.optimize();

    for (auto dnode : outputs_to_recalc)
        dnode->recalc_output_layout();
}

void program_impl::post_optimize_weights(layout_optimizer& lo)
{
    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_weights = [this, &lo](program_node const& weights, auto& node, size_t dep_idx)
    {
        auto wtype = weights.type();
        auto* impl = node.get_selected_impl().get();
        auto output_layout = node.get_output_layout();
        const auto weights_type = layout_optimizer::data_type::weights;
        if (wtype == data::type_id())
        {
            lo.add_weights_for_optimization(
                impl->_kernel_data.weightsReorderParams,
                weights.as<data>().typed_desc(),
                weights_type);
        }
        else if (wtype == input_layout::type_id())
        {
            auto reorders = lo.get_generic_layer(
                impl->_kernel_data.weightsReorderParams,
                weights.as<input_layout>().typed_desc()->id,
                output_layout,
                weights_type);

            for (auto& reorder : reorders)
            {
                this->add_intermediate(reorder.first, node, dep_idx);
            }
        }
    };

    //generic lambda function which prepares given primitive for weights optimization
    //it deduces the type of weights from the type of the argument and calls 'add_weights' for all
    //weights used by given primitive.
    //argument should match few requirements:
    // - it should be of a form 'typed_program_node<T>&'
    // - 'T.weights' should be either of type 'primitive_id' or 'std::vector<primitive_id>'
    const auto prep_opt = [this, &add_weights](auto& node) -> void
    {
        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
        for (auto i = weights_offset; i < bias_offset; i++)
        {
            add_weights(node.get_dependency(i), node, i);
        }
    };

    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;
        if (prim.type() == convolution::type_id())
        {
            prep_opt(prim.as<convolution>());
        }
        else if (prim.type() == deconvolution::type_id())
        {
            prep_opt(prim.as<deconvolution>());
        }
        else if (prim.type() == fully_connected::type_id())
        {
            prep_opt(prim.as<fully_connected>());
        }
    }

    //all optimizing primitives has been added and inputs for all primitives has been updated.
    //run reorders now
    lo.optimize();
}

void program_impl::apply_needed_padding(program_node& node, program_node& prev_node,
                                                const padding& needed_padding)
{
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>())
    {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_" + prev_node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program_impl::prepare_padding()
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
                    prim->output_size, filter_size, prim->input_offset, prim->stride, {1, 1, 1, 1}, true, 1);

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
        if (pair.second->get_primitive()->type != convolution::type_id())
            continue;

        auto& node = pair.second->as<convolution>();
        if (node.get_dependencies().empty())
            continue;

        auto conv = node.get_primitive();
        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx)
        {
            continue;
        }

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

void program_impl::prepare_buffer_fusing()
{
    for (auto& node : processing_order)
    {
        do_for_types<concatenation>(*node, [this](concatenation_node& node)
        {
            auto format = node.get_output_layout().format;
            if (format != format::bfyx)
                return;

            //if any of this node's inputs is used by more than one primitive do not fuse buffers
            // todo: in future, if this case is problem, it can be optimized further to enable buffer fusing
            //       per single input rather than all/none
            // + restrict input types to pooling, convolution and activation only due to problems with output padding on b and f
            for (auto const& input : node.get_dependencies())
                if (input->get_users().size() > 1 || (!input->is_type<pooling>() && !input->is_type<convolution>() && !input->is_type<activation>()))
                    return;

            // buffer fusing should not be performed if one of inputs produces padded output since
            // it could break desired memory alignment. On the other hand, if this node uses all inputs
            // exclusively (see check above) they should not have output padding set since concatenation
            // does not ask for any.
            assert(!node.has_padded_dependency());
            if (node.has_padded_dependency())
                return;

            auto concat_axis = node.get_primitive()->axis;
            auto padd = node.get_output_layout().data_padding;

            //calculate lower and upper paddding so they sum up to the buffer size
            // at the beginning lower padd points to the starting position of the output data
            //
            //   |--- lower padd ---| ------------------ upper padd -----------------------|
            //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
            tensor lower_padd = padd.lower_size();
            tensor upper_padd = padd.upper_size();

            upper_padd.raw[concat_axis] = node.get_output_layout().get_buffer_size().raw[concat_axis] - lower_padd.raw[concat_axis];

            for (auto const& input : node.get_dependencies())
            {
                auto input_lenght = input->get_output_layout().size.raw[concat_axis];

                // shrink upper pad so it points at the end of the input's buffer
                //
                //   |--- lower padd ---|                    |---------- upper padd -----------|
                //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
                upper_padd.raw[concat_axis] -= input_lenght;

                // set new padding for input
                input->set_output_padding(padding(lower_padd.sizes(), upper_padd.sizes()));

                // move lower padd further
                //
                //   |-------------- lower padd -------------|---------- upper padd -----------|
                //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|

                lower_padd.raw[concat_axis] += input_lenght;
            }
            
            node.can_be_optimized(true);
        });

        // zero copy 
        do_for_types<crop>(*node, [this](crop_node& node)
        {
            if (node.get_dependencies().size() == 1 &&
                node.get_users().size() > 0)
            {
                // optimization is avaiable for croping across depth(features) only
                // if output padding has defined padding accross featuers already it wouldn't 
                // work because it expect to have zeros in the padded area.
                auto format = node.get_output_layout().format;
                auto crop_prim = node.get_primitive();
                auto input_layout = node.get_dependency(0).get_output_layout();
                auto in_place_layout = node.get_output_layout();
                auto out_padd = node.get_output_layout().data_padding;
                if (format == format::bfyx &&
                    crop_prim->reference_input.batch[0] == input_layout.size.batch[0] &&
                    crop_prim->reference_input.spatial[0] == input_layout.size.spatial[0] &&
                    crop_prim->reference_input.spatial[1] == input_layout.size.spatial[1] &&
                    out_padd.lower_size().feature[0] == 0 &&
                    out_padd.upper_size().feature[0] == 0)
                {
                    //  Regular crop
                    //  crop input buffer
                    //  |___________data____________|
                    //  
                    //  crop output buffer
                    //  |-------->| offsets[f]  |<--|
                    //            |_____data____|
                    //             <------------>
                    //           reference size
                    //
                    //  Inplace crop
                    //  crop output buffer
                    //  |_low_pad_|__data_size__|___|<-upper pad

                    node.set_output_padding(padding(
                    { out_padd.lower_size().batch[0], crop_prim->offsets.feature[0], out_padd.lower_size().spatial[0], out_padd.lower_size().spatial[1] },
                    { out_padd.upper_size().batch[0], in_place_layout.size.feature[0] - crop_prim->offsets.feature[0] - crop_prim->reference_input.feature[0],
                        out_padd.upper_size().spatial[0], out_padd.upper_size().spatial[1] }));
                    node.can_be_optimized(true);
                }
            }
        });
    }
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

void program_impl::add_intermediate(program_node& node, program_node& next, size_t prev_idx)
{
    auto& prev = next.get_dependency(prev_idx);
    //firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    add_connection(prev, node);
    next.replace_dependency(prev_idx, node);
    node.processing_itr = processing_order.insert(next.processing_itr, &node);
}

void program_impl::remove_if_dangling(program_node& node)
{
    if (!node.users.empty() || !node.dependencies.empty() || node.is_output())
        return;

    processing_order.erase(node.processing_itr);
    optimized_out.push_back(node.id());
    nodes_map.erase(node.id());
}

void program_impl::calc_processing_order()
{
    processing_order.clear();

    //run dfs to sort nodes topologically
    for (auto input : inputs)
    {
        if (input->is_marked())
            continue;

        input->mark();
        std::list<std::pair<program_node*, std::list<program_node*>::const_iterator>> stack = { std::make_pair(input, input->users.begin()) };

        while (!stack.empty()) //imitate call stack
        {
        new_frame:
            auto& frame = stack.back();

            while (frame.second != frame.first->users.end())
            {
                auto successor = *frame.second;
                ++frame.second;

                if (!successor->is_marked())
                {
                    successor->mark();

                    //recurrence call
                    stack.push_back(std::make_pair(successor, successor->users.begin()));
                    goto new_frame;
                }
            }

            //we have finished processing one node so add it to the processing queue
            processing_order.push_front(frame.first);
            frame.first->processing_itr = processing_order.begin();

            //return from call
            stack.pop_back();
        }
    }

    for (auto& node : nodes_map)
        node.second->unmark();
}

void program_impl::forward_bfs(std::function<void(program_node&)> const& mark_func, std::function<void(program_node&)> const& unmark_func) const
{
    if (!mark_func && !unmark_func)
        return;

    std::list<const std::list<program_node*>*> stack = { &inputs };
    while (!stack.empty())
    {
        auto nodes_list = stack.front();
        stack.pop_front();

        for (auto node : *nodes_list)
        {
            if (!node->is_marked())
            {
                node->mark();
                if (mark_func)
                    mark_func(*node);
                if (!node->get_users().empty())
                    stack.push_back(&node->get_users());
            }
        }
    }

    for (auto& node : nodes_map)
    {
        if (unmark_func)
            unmark_func(*node.second);
        node.second->unmark();
    }
}

void program_impl::backward_bfs(std::function<void(program_node&)> const& mark_func, std::function<void(program_node&)> const& unmark_func) const
{
    if (!mark_func && !unmark_func)
        return;

    std::list<const std::vector<program_node*>*> stack = { &outputs };
    while (!stack.empty())
    {
        auto nodes_list = stack.front();
        stack.pop_front();

        for (auto node : *nodes_list)
        {
            if (!node->is_marked())
            {
                node->mark();
                if (mark_func)
                    mark_func(*node);
                if (!node->get_dependencies().empty())
                    stack.push_back(&node->get_dependencies());
            }
        }
    }

    for (auto& node : nodes_map)
    {
        if (unmark_func)
            unmark_func(*node.second);
        node.second->unmark();
    }
}

}