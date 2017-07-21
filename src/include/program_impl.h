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
#pragma once
#include "api/CPP/program.hpp"

#include "refcounted_obj.h"
#include "topology_impl.h"
#include "engine_impl.h"
#include "primitive_type.h"
#include "meta_utils.h"

#include <list>
#include <algorithm>

namespace cldnn
{

struct program_impl;
struct primitive_impl;
struct engine_impl;
class layout_optimizer;

template <class T>
struct typed_program_node;

/*
    Base class for all primitives which wraps API class and extends it to be used
    in graph context.

    Besides primitive description provided by user, this class includes functionality to
    ask for direct predecessors and succesors as well as takes care of changes to primitive
    which would affect other graph's nodes (the most commont case is probably calculating output layout).

    At graph level, all connections between nodes are directly stored inside program_nodes - in oposite
    to API level where all primitives store only ids of related ones.
*/
struct program_node
{
    friend struct program_impl;

    program_node(std::shared_ptr<primitive> prim, program_impl& prog) : desc(prim), myprog(prog)
    {
        output_layout.data_padding = prim->output_padding;
    }

public:
    auto id() const { return desc->id; }
    auto type() const { return desc->type; }

    template <class PType>
    bool is_type() const
    {
        static_assert(meta::is_primitive_v<PType>, "Type argument for program_node::is_type should be a non-const, non-volatile type derived from primitive");
        return type() == PType::type_id();
    }

    auto& get_program() { return myprog; }
    auto const& get_program() const { return myprog; }

    auto get_selected_impl() const { return selected_impl; }

    auto const& get_dependencies() const { return dependencies; }
    auto& get_dependency(size_t idx) const { return *dependencies.at(idx); }

    void replace_dependency(size_t idx, program_node& new_dep);
    void replace_dependency(program_node const& old_dep, program_node& new_dep);

    auto const& get_users() { return users; }
    // for const method, add const to stored successors/predecessors
    auto const& get_users() const { return reinterpret_cast<const std::list<const program_node*>&>(users); }

    //do not modify primitive directly to keep synchronisation wit graph
    std::shared_ptr<const primitive> get_primitive() const { return desc; }

    //primitive modification functions
    void set_output_padding(padding const& padd)
    {
        //changing output padding shouldn't cause any changes to other primitives
        //so just change it
        output_layout.data_padding = padd;
    }

    void merge_output_padding(padding const& padd)
    {
        set_output_padding(padding::max(padd, output_layout.data_padding));
    }

    layout get_output_layout();

    layout get_output_layout() const
    {
        if (!valid_output_layout)
            throw std::runtime_error("Output layout not calculated");

        return output_layout;
    }

    void recalc_output_layout()
    {
        valid_output_layout = false;
        get_output_layout();
    }

    bool is_padded() { return static_cast<bool>(get_output_layout().data_padding); }
    bool is_padded() const { return static_cast<bool>(get_output_layout().data_padding); }

    bool has_padded_dependency() const
    {
        return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const program_node* node) { return node->is_padded(); });
    }

    auto is_input() const { return dependencies.empty(); }
    auto is_endpoint() const { return users.empty(); }
    auto set_output(bool out) { output = out; }
    auto is_output() const { return output; }

    auto mark() { user_mark = true; }
    auto unmark() { user_mark = false; }
    auto is_marked() const { return user_mark; }

    //conversion from generic to specific
    template <class To, class..., class = std::enable_if_t<!std::is_same<To, primitive>::value>>
    typed_program_node<To>& as()
    {
        if (get_primitive()->type != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To>&>(*this);
    }

    template <class To, class..., class = std::enable_if_t<!std::is_same<To, primitive>::value>>
    typed_program_node<To> const& as() const
    {
        if (get_primitive()->type != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To> const&>(*this);
    }

    template <class To>
    operator typed_program_node<To>& ()
    {
        return as<To>();
    }

    template <class To>
    operator typed_program_node<To> const& () const
    {
        return as<To>();
    }

protected:
    std::shared_ptr<primitive> desc;
    program_impl& myprog;

    std::shared_ptr<primitive_impl> selected_impl;

    bool valid_output_layout = false;
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());

    std::vector<program_node*> dependencies;
    std::list<program_node*> users;

    std::list<program_node*>::const_iterator processing_itr;

    bool output = false;
    bool user_mark = false;

    void invalidate_users() const
    {
        for (auto& user : users)
        {
            if (user->valid_output_layout)
            {
                user->valid_output_layout = false;
                user->invalidate_users();
            }
        }
    }
};

/*
    Template class used to indicate that usage context requires 'program_node' to wrap primitive
    of type 'PType'. Successful conversion from 'program_node' to 'typed_program_node<PType>' means
    that this restriction in fact holds and functions/method/etc. may saftly use uderlaying primitive.

    This class shadows 'get_primitive' method from base class which now returns pointer to more specific
    type.
*/
template <class PType>
struct typed_program_node_base : public program_node
{
    static_assert(meta::is_primitive_v<PType>, "PType should be a non-const, non-volatile class derived from primitive");

    friend struct program_impl;

public:
    using program_node::program_node;

    std::shared_ptr<const PType> get_primitive() const { return std::static_pointer_cast<const PType>(program_node::get_primitive()); }

protected:
    std::shared_ptr<PType> typed_desc() const { return std::static_pointer_cast<PType>(desc); }
};

/*
    Actual template class used in context which requires 'program_node' to wrap
    primitive of type 'PType'. This class is introduced to provide possibility of explicit specialization.
    In most cases such specializations would add accessors to make access to PType-specific fields easier.

    It's not required to specialize this class for new primitives types.
*/
template <class PType>
struct typed_program_node : public typed_program_node_base<PType>
{
    using typed_program_node_base<PType>::typed_program_node_base;

    auto& input() const { return program_node::get_dependency(0); }
};

/*
    cldnn_program implementation
*/
struct program_impl : public refcounted_obj<program_impl>
{
    friend struct program_node;

public:
    program_impl(engine_impl::ptr engine, topology_impl const& topology, build_options const& options);

    auto get_engine() const { return engine; }
    auto get_options() const { return options; }

    std::list<std::shared_ptr<program_node>> get_nodes() const;

    auto& get_node(primitive_id const& id) 
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
    auto const& get_node(primitive_id const& id) const
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

private:
    engine_impl::ptr engine;
    build_options options;

    std::list<program_node*> inputs;
    std::vector<program_node*> outputs;
    std::list<program_node*> processing_order;

    std::map<primitive_id, std::shared_ptr<program_node>> nodes_map;

    std::list<primitive_id> optimized_out;

    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    bool output_size_handling_enabled;

    /*
    ** High-level functions, in order of usage
    */
    void init_graph(topology_impl const& topology);
    void pre_optimize_graph();
    void post_optimize_graph();
    void compile_graph();

    void set_outputs();

    /*
    ** Analysis functions
    */
    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    void analyze_output_size_handling_need();

    /*
    ** Optimization functions
    */
    void trim_to_outputs();
    void reorder_inputs(layout_optimizer& lo);
    void pre_optimize_bias(layout_optimizer& lo);
    void post_optimize_weights(layout_optimizer& lo);
    void apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding);
    void prepare_padding();
    void prepare_buffer_fusing();

    /*
    ** Utilities
    */

    //returns already existing program_node for given primitive 'prim' (lookup in 'nodes_map')
    //if it was previously created, otherwise creates and then returns program_node
    program_node& get_or_create(std::shared_ptr<primitive> prim);

    // Inserts given program_node 'node' as an intermediate node between 'next' and it's
    //  dependency at 'prev_idx' index.
    void add_intermediate(program_node& prim, program_node& next, size_t prev_idx);

    // Gets or creates program_node for given primitive 'prim' and inserts it as an intermediate
    // node between 'next' and it's dependency at 'prev_idx' index.
    void add_intermediate(std::shared_ptr<primitive> prim, program_node& next, size_t prev_idx)
    {
        add_intermediate(get_or_create(prim), next, prev_idx);
    }

    void add_connection(program_node& prev, program_node& next)
    {
        prev.users.push_back(&next);
        next.dependencies.push_back(&prev);
    }

    void remove_if_dangling(program_node& node);

    void calc_processing_order();

    void forward_bfs(std::function<void(program_node&)> const& mark_func = nullptr, std::function<void(program_node&)> const& unmark_func = nullptr) const;
    void backward_bfs(std::function<void(program_node&)> const& mark_func = nullptr, std::function<void(program_node&)> const& unmark_func = nullptr) const;
};
}

API_CAST(::cldnn_program, cldnn::program_impl)
