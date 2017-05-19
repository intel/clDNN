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
#include "api/CPP/convolution.hpp"
#include "api/CPP/deconvolution.hpp"
#include "api/CPP/data.hpp"
#include "api/CPP/eltwise.hpp"
#include "api/CPP/input_layout.hpp"
#include "api/CPP/pooling.hpp"
#include "api/CPP/reorder.hpp"

#include "convolution_inst.h"


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
    : engine(engine), options(options)
{
    init_graph(topology);
    optimize_graph();
    compile_graph();
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

void program_impl::optimize_graph()
{
    trim_to_outputs();

    if (options.get<build_option_type::optimize_data>()->enabled())
    {
        layout_optimizer lo(engine);
        reorder_inputs(lo);
        optimize_weights(lo);
    }

    prepare_padding();
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
        }
    });

    for (auto const& opt : optimized_out)
        nodes_map.erase(opt);
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
        }

        if (new_input)
            add_intermediate(new_input, conv_node, 0);
    };

    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;

        //there's an assumption that only convolution will take data/input_layout as input
        //exception to that rule would be a convolution which takes a reorder as input - see reoder_input above
        do_for_types<convolution>(prim,
            reorder_input       //case for convolution
            );
    }
}

void program_impl::optimize_weights(layout_optimizer& lo)
{
    std::list<program_node*> outputs_to_recalc;

    //lambda function which finds weights primitive with given pimitive_id and adds it to weights_optimizer
    //this function is reused in all cases (convolution weights, convolution bias, fc weights and fc bias) and does
    //some basic sanity checks about existence of the primitive and it's type. throws std::logic_error
    const auto add_weights = [this, &lo, &outputs_to_recalc](program_node& weights, layout_optimizer::data_type weights_type, auto& node, layout const& output_layout, size_t dep_idx)
    {
        if (weights.type() == data::type_id())
        {
            lo.add_weights_for_optimization(weights.as<data>().typed_desc(), weights_type,
                node.get_primitive(), output_layout);
            outputs_to_recalc.push_back(&weights);
        }
        else if (weights.type() == input_layout::type_id())
        {
            auto reorder = lo.get_reorder(
                weights.as<input_layout>().typed_desc()->layout,
                weights.id(),
                weights_type,
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
    const auto prep_opt = [this, &add_weights](auto& node) -> void
    {
        auto output_layout = node.get_output_layout();

        auto weights_offset = node.get_primitive()->input.size();
        auto bias_offset = weights_offset + wrap_if_single(node.get_primitive()->weights).size();
        auto i = weights_offset;
        while (i < node.get_dependencies().size())
        {
            auto data_type = i < bias_offset ? layout_optimizer::data_type::weights : layout_optimizer::data_type::bias;
            add_weights(node.get_dependency(i), data_type, node, output_layout, i);
            ++i;
        }
    };

    for (auto& p : nodes_map)
    {
        auto& prim = *p.second;

        do_for_types<convolution, fully_connected, deconvolution>(prim,
            prep_opt,   //case for convolution
            prep_opt,   //case for fully_connected
            prep_opt    //case for deconvolution
            );
    }

    //all optimizing primitives has been added and inputs for all primitives has been updated.
    //run reorders and replace cldnn::data::mem
    lo.optimize();

    for (auto dnode : outputs_to_recalc)
        dnode->recalc_output_layout();
}

void program_impl::prepare_padding()
{
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

        conv_input_node.merge_output_padding(needed_padding);
    }
}

program_node& program_impl::get_or_create(std::shared_ptr<primitive> prim)
{
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = std::make_shared<program_node>(prim, *this);
    nodes_map.insert(itr, { prim->id, new_node });
    return *new_node;
}

void program_impl::add_intermediate(program_node& node, program_node& next, size_t prev_idx)
{
    auto& prev = next.get_dependency(prev_idx);
    //firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    add_connection(prev, node);
    next.replace_dependency(prev_idx, node);
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