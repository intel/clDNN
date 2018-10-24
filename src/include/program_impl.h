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
#include "engine_impl.h"

#include <list>

namespace cldnn
{

struct topology_impl;
struct primitive_impl;
struct program_node;

class layout_optimizer;
class constants_propagator;
/*
    cldnn_program implementation
*/
struct program_impl : public refcounted_obj<program_impl>
{
    friend struct program_node;
    friend class trim_to_outputs;   // to be removed when possible
    friend class prepare_buffer_fusing; // to be removed when possible
    friend class prepare_primitive_fusing; // to be removed when possible
    friend class propagate_constants; // to be removed when possible
    friend class reorder_inputs;  // to be removed when possible
    friend class post_optimize_weights; // to be removed when possible
    friend class remove_redundant_reorders; // to be removed when possible, remove_redundant_reorders uses extract_and_remove which is private
    friend class pre_optimize_bias; // to be removed when possible
    friend class prepare_depthwise_sep_opt; // to be removed when possible
    friend class prep_opt_depthwise_sep_post; // to be removed when possible
    friend class eltwise_shrinking; // to be removed when possible
    friend class eltwise_remove_stride; // to be removed when possible

public:
    struct nodes_ordering
    {
    public:
        typedef std::list<program_node*> list_of_nodes;
        typedef list_of_nodes::const_iterator const_iterator;
        const_iterator begin() const { return _processing_order.begin(); }
        const_iterator end() const { return _processing_order.end(); }
        const_iterator get_processing_iterator(program_node& node);
        void calc_processing_order_visit(program_node* node);
        void calc_processing_order(program_impl &p);
        void update_processing_numbers();
        void calculate_BFS_processing_order();
        size_t size() { return _processing_order.size(); }
        bool is_correct(program_node* node);
        void clear();
        void erase(const_iterator i);
        program_impl::nodes_ordering::const_iterator insert(const_iterator i, program_node* node);

    private:
        list_of_nodes _processing_order;
        std::map<program_node*, const_iterator> processing_order_iterators;
    };

    template <class T>
    struct single_element_container
    {
        single_element_container(T& t) : elem(&t)
        {}
        constexpr size_t size() const { return 1; }
        single_element_container begin() const { return single_element_container(elem); }
        single_element_container end() const { return single_element_container(nullptr); }
        single_element_container& operator ++() { elem = nullptr; return *this; }
        bool operator !=(single_element_container const& sec) { return elem != sec.elem; }

       T operator *() { return *elem; }

    private:
        single_element_container(T* t) : elem(t)
        {}

        T* elem;
    };

    program_impl(engine_impl& engine_ref, topology_impl const& topology, build_options const& options, bool is_internal);
    engine_impl& get_engine() const { return *engine; }
    const build_options& get_options() const { return options; }
    std::list<program_node*>& get_inputs() { return inputs; }     // ToDo: redesign trim to ouptut pass to make it const as_well as get_engine and get options 
    std::vector<program_node*>& get_outputs() { return outputs; }  // ToDo: redesign reorder-inputs pass to make it const as_well as get_engine and get options 
    bool is_debug_build() const { return options.get<build_option_type::debug>()->enabled(); }
    std::list<std::shared_ptr<program_node>> get_nodes() const;
    const nodes_ordering& get_processing_order() const;
    nodes_ordering& get_processing_order();
    const std::list<primitive_id>& get_optimized_out() const { return optimized_out; }
    bool has_node(const primitive_id& prim) const { return nodes_map.count(prim) > 0; }
    program_node& get_node(primitive_id const& id);
    program_node const& get_node(primitive_id const& id) const;
    void dump_memory_pool() const;

private:
    uint32_t prog_id = 0;

    engine_impl::ptr engine;
    build_options options;

    std::list<program_node*> inputs;
    std::vector<program_node*> outputs;
    nodes_ordering processing_order;

    std::map<primitive_id, std::shared_ptr<program_node>> nodes_map;
    std::list<primitive_id> optimized_out;

    /*
    ** High-level functions, in order of usage
    */
    void init_graph(topology_impl const& topology);
    void pre_optimize_graph();
    void post_optimize_graph();
    void compile_graph();
    void cleanup();

    /*
    ** Initialization functions
    */
    void set_outputs();
    void calc_prior_boxes();

    /*
    ** Analysis functions
    */
    void mark_constants();
    void mark_data_flow();
    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    bool analyze_output_size_handling_need();
    void replace_nodes_pre();
    void replace_nodes_post();
	void handle_lstm();
    void handle_reshape();

    /*
    ** Optimization functions
    */
    void apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding);
    void prepare_padding(bool output_size_handling_enabled);

    /*
    ** Memory pool functions
    */
    void prepare_memory_dependencies();
    void basic_memory_dependencies();
    void skipped_branch_memory_dependencies();
    void oooq_memory_dependencies();
    std::string get_memory_dependencies_string() const;

    /*
    ** Utilities
    */

    //Reverses connection - user becomes dependency.
    void reverse_connection(program_node& dep_node, program_node& user_node);

    //returns already existing program_node for given primitive 'prim' (lookup in 'nodes_map')
    //if it was previously created, otherwise creates and then returns program_node
    program_node& get_or_create(std::shared_ptr<primitive> prim);

    // Inserts given program_node 'node' as an intermediate node between 'next' and it's
    //  dependency at 'prev_idx' index.
    void add_intermediate(program_node& node, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep = true);

    // Gets or creates program_node for given primitive 'prim' and inserts it as an intermediate
    // node between 'next' and it's dependency at 'prev_idx' index.
    void add_intermediate(std::shared_ptr<primitive> prim, program_node& next, size_t prev_idx, bool connect_int_node_with_old_dep = true);

    void add_connection(program_node& prev, program_node& next);

    void remove_connection(program_node& prev, program_node& next);

    void remove_all_connections(program_node& node);

    void rename(program_node & node, primitive_id const & new_id);
    void swap_names(program_node& node1, program_node& node2);
    void replace_all_usages(program_node& old_node, program_node& new_node);

    //old_node - node which will be replaced
    //new_node - node which will replace the old one
    //replace_whole_branch - if set to true, 'old_node' will be replaced with all its dependencies and new_node will retain its dependencies
    //  old's dependencies which are post-dominates by 'old_node' will also be removed
    void replace(program_node& old_node, program_node& new_node, bool replace_whole_branch, bool check_output_layouts_integrity = true);

    //returns if 'node' has been removed
    bool remove_if_dangling(program_node& node, bool detach_whole_branch = false);

    //removes a node from the graph and deletes it afterwards,
    //prereq: node cannot be marked as output and has to have exactly one dependency
    //returns if 'node' has been extracted and removed successfully
    bool extract_and_remove(program_node& node);
    void dump_program(const char* stage, bool with_full_info, std::function<bool(program_node const&)> const& filter = nullptr) const;
    //Dumps weights and biasses in serialization process, not working yet, in progress.
    void dump_weights_and_biasses(std::vector<unsigned long long>& offsets, std::vector<std::string>& data_names, std::ofstream& file_stream) const;
    //Makes serialization with given name.
    //Placeholder, not working yet, in progress.
    void serialize(std::string network_name, std::function<bool(program_node const&)> const& filter = nullptr) const;
};

}

API_CAST(::cldnn_program, cldnn::program_impl)
