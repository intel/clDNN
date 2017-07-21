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

#include <iterator>
#include "kernel.h"
#include "memory_gpu.h"

namespace neural { namespace gpu {

    namespace
    {
        class memory_arg {
            cldnn::memory _mem;

        protected:
            memory_arg(const cldnn::memory& mem) : _mem(mem) {}

        public:
            const cl::Buffer& get_buffer() const { return static_cast<const gpu_buffer*>(api_cast(_mem.get()))->get_buffer(); }
        };

        class input_mem : public memory_arg {
        public:
            input_mem(const cldnn::memory& mem) :memory_arg(mem) {}
        };

        class output_mem : public memory_arg {
        public:
            output_mem(const cldnn::memory& mem) :memory_arg(mem) {}
        };

        template<typename T, class Enable = void>
        struct kernel_arg_handler;

        template<typename T>
        struct kernel_arg_handler<T, typename std::enable_if<!std::is_base_of<memory_arg, T>::value>::type> {
            static const T& get(const T& arg) { return arg; }
        };

        template<typename T>
        struct kernel_arg_handler<T, typename std::enable_if<std::is_base_of<memory_arg, T>::value>::type> {
            static const cl::Buffer& get(const T& arg) { return arg.get_buffer(); }
        };

        inline cl::NDRange toNDRange(const std::vector<size_t>& v)
        {
            switch (v.size())
            {
            case 1:
                return cl::NDRange(v[0]);
            case 2:
                return cl::NDRange(v[0], v[1]);
            case 3:
                return cl::NDRange(v[0], v[1], v[2]);
            default:
                return cl::NullRange;
            }
        }

        void set_arguments(
            cl::Kernel& kernel,
            const kernel_selector::kernel_arguments& args,
            const kernel::kernel_arguments_data& data)
        {
            for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++)
            {
                cl_int status = CL_INVALID_ARG_VALUE;

                switch (args[i].t)
                {
                case kernel_selector::kernel_argument_types::INPUT:
                    if (args[i].index < data.inputs.size() && data.inputs[args[i].index])
                    {
                        const auto& input_mem = data.inputs[args[i].index];
                        if (input_mem)
                        {
                            status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*input_mem));
                        }
                    }
                    break;
                case kernel_selector::kernel_argument_types::OUTPUT:
                    if (data.output)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::output_mem>::get(*data.output));
                    }
                    break;
                case kernel_selector::kernel_argument_types::WEIGHTS:
                    if (data.weights)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*data.weights));
                    }
                    break;
                case kernel_selector::kernel_argument_types::BIAS:
                    if (data.bias)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*data.bias));
                    }
                    break;
                case kernel_selector::kernel_argument_types::LOOKUP_TABLE:
                    if (data.lookup_table)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*data.lookup_table));
                    }
                    break;
                case kernel_selector::kernel_argument_types::SCALE_TABLE:
                    if (data.scale_table)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*data.scale_table));
                    }
                    break;
                case kernel_selector::kernel_argument_types::SLOPE:
                    if (data.slope)
                    {
                        status = kernel.setArg(i, kernel_arg_handler<gpu::input_mem>::get(*data.slope));
                    }
                    break;
                case kernel_selector::kernel_argument_types::SPLIT:
                    status = kernel.setArg(i, data.split);
                    break;
                case kernel_selector::kernel_argument_types::SCALAR:
                    if (data.scalars && args[i].index < data.scalars->size())
                    {
                        const auto& scalar = (*data.scalars)[args[i].index];
                        switch (scalar.t)
                        {
                        case kernel_selector::kernel_scalar_argument_types::UINT8:
                            status = kernel.setArg(i, scalar.v.u8);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::UINT16:
                            status = kernel.setArg(i, scalar.v.u16);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::UINT32:
                            status = kernel.setArg(i, scalar.v.u32);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::UINT64:
                            status = kernel.setArg(i, scalar.v.u64);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::INT8:
                            status = kernel.setArg(i, scalar.v.s8);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::INT16:
                            status = kernel.setArg(i, scalar.v.s16);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::INT32:
                            status = kernel.setArg(i, scalar.v.s32);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::INT64:
                            status = kernel.setArg(i, scalar.v.s64);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::FLOAT32:
                            status = kernel.setArg(i, scalar.v.f32);
                            break;
                        case kernel_selector::kernel_scalar_argument_types::FLOAT64:
                            status = kernel.setArg(i, scalar.v.f64);
                            break;
                        default:
                            break;
                        }
                    }
                default:
                    break;
                }

                if (status != CL_SUCCESS)
                {
                    throw std::runtime_error("Error set args\n");
                }
            }
        }
    }

    cldnn::refcounted_obj_ptr<cldnn::event_impl> kernel::run(
        const kernel_selector::cl_kernel_data& kernel_data,
        const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& dependencies,
        const kernel_arguments_data& args) const
    {
        cl::Event end_event;
        std::vector<cl::Event> events;

        bool run_this_layer = false;
        if (context()->enabled_single_kernel())
        {
            std::string proper_layer_name = kernel_data.layerID;
            if (proper_layer_name.compare(context()->single_kernel_name()) == 0)
            {
                run_this_layer = true;
            }
        }
        else
        {
            run_this_layer = true;

            events.reserve(dependencies.size());
            for (auto& dependency : dependencies)
            {
                events.emplace_back(dependency->get());
            }
        }

        if (run_this_layer)
        {
            auto clkernel = context()->get_kernels_cache().get_kernel(_kernel_id);

            set_arguments(clkernel, kernel_data.arguments, args);

            context()->queue().enqueueNDRangeKernel(
                clkernel,
                cl::NullRange,
                toNDRange(kernel_data.workGroups.global),
                toNDRange(kernel_data.workGroups.local),
                &events,
                &end_event);
        }

        return{ new cldnn::event_impl(end_event), false };
    }

} }