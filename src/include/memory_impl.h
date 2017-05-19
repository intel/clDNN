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

#include "api_impl.h"
#include "engine_impl.h"
#include "refcounted_obj.h"

namespace cldnn
{

struct memory_impl : refcounted_obj<memory_impl>
{
    memory_impl(const engine_impl::ptr& engine, layout layout): _engine(engine), _layout(layout){}
    virtual ~memory_impl() = default;
    virtual void* lock() = 0;
    virtual void unlock() = 0;
    size_t size() const { return _layout.bytes_count(); }
    virtual bool is_allocated_by(const refcounted_obj_ptr<engine_impl>& engine) const { return engine == _engine; }
    const refcounted_obj_ptr<engine_impl>& get_engine() const { return _engine; }
    const layout& get_layout() const { return _layout; }
protected:
    const engine_impl::ptr _engine;
    const layout _layout;
};

struct simple_attached_memory : memory_impl
{
    simple_attached_memory(layout layout, void* pointer)
        : memory_impl(nullptr, layout), _pointer(pointer)
    {
    }

    void* lock() override { return _pointer; }
    void unlock() override {}
private:
    void* _pointer;
};

}

API_CAST(::cldnn_memory, cldnn::memory_impl)
