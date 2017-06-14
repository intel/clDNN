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

#include "api/CPP/memory.hpp"
#include "api/CPP/primitive.hpp"

#include "event_impl.h"
#include "program_impl.h"
#include "meta_utils.h"

#include <memory>
#include <vector>
#include <boost/optional.hpp>

namespace neural { namespace gpu { class gpu_toolkit; } }

namespace cldnn
{

struct network_impl;
class primitive_inst;

template <class PType>
class typed_primitive_inst;

/*
    Base class for all implementations.
*/
struct primitive_impl
{
    virtual ~primitive_impl() = default;

    virtual event_impl::ptr execute(const std::vector<event_impl::ptr>& events, primitive_inst& instance) = 0;
};

/*
    Base class for all primitive instances.
    It's main responsibility is to allocate memory required to run single, specified in ctor,
    program_node. It also contains informations about it's predecessor in network graph and checks (<<-- TODO)
    if output should be recalculated between network runs.
*/
class primitive_inst
{
    template <class PType>
    friend class typed_primitive_inst;

public:
    virtual ~primitive_inst() = default;

    const std::vector<std::shared_ptr<const primitive_inst>>& dependencies() const
    { 
        return reinterpret_cast<std::vector<std::shared_ptr<const primitive_inst>> const&>(_deps);
    }

    const memory& dep_memory(size_t index) const { return dependencies().at(index)->output_memory(); }
    const memory& output_memory() const { return _output.get(); }
    primitive_type_id type() const { return _node.get_primitive()->type; }
    primitive_id id() const { return _node.get_primitive()->id; }
    const auto desc() const { return _node.get_primitive(); }
    network_impl& get_network() const { return _network; }

    event_impl::ptr execute(const std::vector<event_impl::ptr>& events);

    auto output_changed() const { return _output_changed; }
    void reset_output_change() { _output_changed = false; }

    //return pointer to const to prevent arbitrary 'execute' call -> use primitive_inst.execute() instead
    const auto get_impl() const { return _impl.get(); }

protected:
    primitive_inst(network_impl& network, program_node const& node, bool allocate_buffer);

    network_impl& _network;
    program_node const& _node;

    std::shared_ptr<primitive_impl> _impl;

    std::vector<std::shared_ptr<primitive_inst>> _deps;

    //_output is optional because its initialization might be postponed (reshape_inst may either allocate it's own buffer or attach input as output
    // depending on reshape_node.is_in_place())
    boost::optional<memory> _output;

    bool _output_changed; //todo: implement output reuse if neither of inputs has changed
    bool _has_valid_input = true; //by default all primitives has valid inputs, exception is input_layout (see input_layout_inst)

    memory allocate_output();

    //event function called by primitive_inst::execute after checking if primitive should rerun and before calling _impl->execute()
    //mainly for reshape (to update output memory if reshape_node.is_in_place() == true)
    virtual void on_execute() {}
};

/*
Base class for all implementation of specified primitive type.
For example, all convolution implementations should derive from typed_primitive_impl<convolution>.
*/
template <class PType>
struct typed_primitive_impl : public primitive_impl
{
    static_assert(meta::is_primitive_v<PType>, "PType should be a non-const, non-volatile class derived from primitive");

private:
    event_impl::ptr execute(const std::vector<refcounted_obj_ptr<event_impl>>& event, primitive_inst& instance) override
    {
        if (instance.type() != PType::type_id())
            throw std::invalid_argument("Implementation type does not match primitive type");
        if (instance.get_impl() != this)
            throw std::invalid_argument("Trying to execute primitive implementation with mismatching primitive instance");

        return execute_impl(event, reinterpret_cast<typed_primitive_inst<PType>&>(instance));
    }

    virtual event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& event, typed_primitive_inst<PType>& instance) = 0;
};


/*
    Base class for all concrete primitive instances.
*/
template<class PType>
class typed_primitive_inst_base : public primitive_inst
{
    static_assert(meta::is_primitive_v<PType>, "PType should be a non-const, non-volatile class derived from primitive");

public:
    using typed_node = typed_program_node<PType>;
    using typed_impl = typed_primitive_impl<PType>;

    const typed_node& node;
    const PType& argument;

    typed_primitive_inst_base(network_impl& network, typed_node const& node)
        : typed_primitive_inst_base(network, node, true)
    {}

protected:
    typed_primitive_inst_base(network_impl& network, typed_node const& node, bool allocate_buffer)
        : primitive_inst(network, node, allocate_buffer)
        , node(_node)
        , argument(*node.get_primitive())
    {}

    typed_primitive_inst_base(network_impl& network, typed_node const& node, memory const& buffer)
        : typed_primitive_inst_base(network, node, false)
    {
        _output = buffer;
    }
};

/*
    Template class which represents instance of primitive 'PType'.
    Each new primitive should explicitly specialize this class.
    The pattern is as follows:
        struct new_primitive {}; // C++ API layer
        template <>
        class typed_primitive_inst<new_primitive> : public typed_primitive_inst_base<new_primitive> {}; // network instance specialization
        using new_primitive_inst = typed_primitive_inst<new_primitive>; //to simplify usage

    Using template specialization instead of dedicated classes for each primitive comes in hand
    when writing other template methods/classes which would like to use primitive_inst.
    As alternative to this, one could use some kind of type traits to translate primitive type
    to related primitive_inst implementation but this approach does the same with less code/classes.
*/
template <class PType>
class typed_primitive_inst : public typed_primitive_inst_base<PType>
{ 
    static_assert(meta::always_false_v<PType>, "Missing typed_primitive_inst specialization");
};

}
