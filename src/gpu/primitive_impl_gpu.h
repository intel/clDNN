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

#pragma once
#include "api/neural.h"
#include "kernel.h"

namespace neural { namespace gpu
{
    struct kernel_arg
    {
        const void* ptr;
        size_t size;
    };

    template<typename T, typename Dummy = std::enable_if<!std::is_base_of<memory_arg, T>::value>>
    class scalar_kernel_arg : public kernel_arg
    {
        T _value;
        scalar_kernel_arg(const T& val) : _value(val)
        {
            ptr = cl::detail::KernelArgumentHandler<T>::ptr(_value);
            size = cl::detail::KernelArgumentHandler<T>::size(_value);
        }
    };

    template<class Mem, typename Dummy = std::enable_if<std::is_base_of<memory_arg, Mem>::value>>
    class memory_kernel_arg : public kernel_arg
    {
        Mem _mem;
        memory_kernel_arg(const neural::memory& arg):_mem(arg)
        {
            ptr = cl::detail::KernelArgumentHandler<cl::Buffer>::ptr(_mem.get_buffer());
            size = cl::detail::KernelArgumentHandler<cl::Buffer>::size(_mem.get_buffer());
        }
    };

    typedef memory_kernel_arg<input_mem> imput_memory_kernel_arg;
    typedef memory_kernel_arg<output_mem> output_memory_kernel_arg;

    template<class Child, class Prim = typename Child::primitive_type>
    struct primitive_impl_gpu :is_an_implementation
    {
        const Prim& _outer;
        gpu::kernel _kernel;
        gpu::kernel_execution_options _exec_options;
        std::vector<std::shared_ptr<kernel_arg>> _kernel_args;

        static void implementation(const void* ptr)
        {
            auto me = static_cast<const Child*>(ptr);
            me->start();
        }

        task_group work() override { return{ { task{ implementation, static_cast<Child*>(this) } }, schedule::single }; }

        static is_an_implementation *create(const Prim& arg) {
            Child::validate(arg);
            return new Child(arg);
        }

    protected:
        explicit primitive_impl_gpu(const type_traits* type_id, const Prim& arg)
            : is_an_implementation(type_id)
            , _outer(arg)
            , _kernel(Child::select_kernel_name(arg), Child::get_jit_constants(arg))
            , _exec_options(Child::get_execution_options(arg))
            , _kernel_args(Child::get_kernel_args(arg))
        {}
    };
} }
