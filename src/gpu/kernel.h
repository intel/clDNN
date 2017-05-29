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
#include "api/CPP/profiling.hpp"
#include "api/CPP/primitive.hpp"

#include "memory_gpu.h"
#include "kernels_cache.h"
#include "event_impl.h"

#include <cmath>
#include <iostream>
#include <sstream>

namespace neural { namespace gpu {

class memory_arg {
    cldnn::memory _mem;

protected:
    memory_arg(const cldnn::memory& mem) : _mem(mem){}

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

// TODO improve to_code_string specializations
template<typename T>
std::string to_code_string(T val) { return std::to_string(val); }

template<>
inline std::string to_code_string<std::string>(std::string val) { return val; }

template<>
inline std::string to_code_string<const char*>(const char* val) { return val; }

template<>
inline std::string to_code_string<char*>(char* val) { return val; }

template<>
inline std::string to_code_string<float>(float val) {
    // 64 chars should be enought to store: "-0x0.123456p-123f /*-0.123456e-123*/"
    char buffer[64] = "";
    if (std::isinf(val))
        std::snprintf(buffer, sizeof(buffer), "%sINFINITY", std::signbit(val) ? "-" : "");
    else
        std::snprintf(buffer, sizeof(buffer), "%.6af /*%.4g*/", double(val), double(val));
    return buffer;
}

template<>
inline std::string to_code_string<double>(double val) {
    // 64 chars should be enought to store: "-0x0.1234567890123p-1234 /*-0.1234567890123e-1074*/"
    char buffer[64] = "";
    if (std::isinf(val))
        std::snprintf(buffer, sizeof(buffer), "%sINFINITY", std::signbit(val) ? "-" : "");
    else
        std::snprintf(buffer, sizeof(buffer), "%.13a /*%.4g*/", val, val);
    return buffer;
}

// TODO refactor jit_constant, make_jit_constant, etc...
class jit_constant {
protected:
    const std::string _name;
    jit_constant(const std::string& name):_name(name){}

public:
    virtual kernels_cache::jit_definitions get_definitions() const = 0;
    virtual ~jit_constant() {}
};

class simple_jit_constant : public jit_constant {
    const std::string _value;

public:
    simple_jit_constant(const std::string& name, const std::string& value)
        :jit_constant(name), _value(value) {}

    kernels_cache::jit_definitions get_definitions() const override {
        return kernels_cache::jit_definitions{ {_name, _value} };
    }
};

template<typename T>
std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, T value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<simple_jit_constant>(name, to_code_string(value)));
}

class vector_jit_constant : public jit_constant {
    const cldnn::tensor _vec;

public:
    vector_jit_constant(const std::string& name, const cldnn::tensor& vec)
        : jit_constant(name), _vec(vec) {}

    kernels_cache::jit_definitions get_definitions() const override {

        kernels_cache::jit_definitions definitions{
            { _name + "_BATCH_NUM", std::to_string(_vec.batch[0]) },
        };

        const char* spatial_names[] = { "X", "Y", "Z", "W" };
        if (_vec.spatial.size() > (sizeof(spatial_names)/sizeof(spatial_names[0])))
            throw std::runtime_error("max 4D images are supported");

        // set default spatial value to "1"
        cldnn::tensor::value_type spatial_value = 1;
        for (size_t i = 0; i < std::max(_vec.spatial.size(), static_cast<size_t>(2)); ++i) {
            // tensor's spatials num is less than 2
            //      then use the value of the last spatial (or default "1")
            if (_vec.spatial.size() > i)
                spatial_value = _vec.spatial[i];
            definitions.emplace_back( _name + "_SIZE_" + spatial_names[i], std::to_string(spatial_value));
        }

        assert(_vec.feature.size() > 0);
        if (_vec.feature.size() > 0) {
            // if number of feature nums is 1 then no suffix
            if(_vec.feature.size() == 1) {
                definitions.emplace_back(_name + "_FEATURE_NUM", std::to_string(_vec.feature[0]));
            }
            else { // else add suffixes
                for (size_t i = 0; i < _vec.feature.size(); ++i) {
                    definitions.emplace_back(_name + "_FEATURE_NUM_" + std::to_string(i), std::to_string(_vec.feature[i]));
                }
            }
        }
        return definitions;
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const cldnn::tensor& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<vector_jit_constant>(name, value));
}

class padding_jit_constant : public jit_constant {
    vector_jit_constant _lower_size_jit;
    vector_jit_constant _upper_size_jit;

public:
    padding_jit_constant(const std::string& name, const cldnn::padding& pad)
        : jit_constant(name),
          _lower_size_jit(name + "_LOWER", pad.lower_size()),
          _upper_size_jit(name + "_UPPER", pad.upper_size()) {}

    kernels_cache::jit_definitions get_definitions() const override {
        auto&& lower_jits = _lower_size_jit.get_definitions();
        auto&& upper_jits = _upper_size_jit.get_definitions();
        lower_jits.insert(lower_jits.cend(), upper_jits.cbegin(), upper_jits.cend());

        return lower_jits;
    }
};

inline std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const cldnn::padding& value) {
    return std::make_shared<padding_jit_constant>(name, value);
}

class memory_jit_constant : public vector_jit_constant {
    const cldnn::memory _mem;

public:
    memory_jit_constant(const std::string& name, const cldnn::memory& mem)
        : vector_jit_constant(name, mem.get_layout().size), _mem(mem){}

    kernels_cache::jit_definitions get_definitions() const override {
        auto result = vector_jit_constant::get_definitions();
        auto data = _mem.pointer<float>();
        std::stringstream ss;
        ss << "(float[]){ ";
        for (size_t i = 0; i < _mem.count(); i++)
            ss << to_code_string(data[i]) << ",";
        ss << " } ";
        result.push_back({ _name, ss.str() });
        return result;
    }
};

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const cldnn::memory& value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<memory_jit_constant>(name, value));
}



class memories_jit_constant : public vector_jit_constant {
    const std::vector<cldnn::memory> _mem;

public:
    memories_jit_constant(const std::string& name, const std::vector<cldnn::memory> mem)
        :vector_jit_constant(name, mem[0].get_layout().size), _mem(mem) {}

    kernels_cache::jit_definitions get_definitions() const override {
        for (size_t i = 1; i < _mem.size(); i++)
        {
            if (_mem[0].count() != _mem[i].count())
                throw std::invalid_argument("All memories must contain the same number of elements!");
        }
        auto result = vector_jit_constant::get_definitions();
        result.push_back({ _name + "_ARRAY_NUM", std::to_string(_mem.size()) });
        std::stringstream ss;
        ss << "(float[][" + std::to_string(_mem[0].count()) + "]) {";
        for (auto& m : _mem)
        {
            auto data = m.pointer<float>();
            ss << "{ ";
            for (size_t i = 0; i < m.count(); i++)
                ss << to_code_string(data[i]) << ",";
            ss << " } ,";
        }
        ss << " } ";
        result.push_back({ _name, ss.str() });
        return result;
    }
};

inline  std::shared_ptr<jit_constant> make_jit_constant(const std::string& name, const std::vector<cldnn::memory> value) {
    return std::static_pointer_cast<jit_constant>(std::make_shared<memories_jit_constant>(name, value));
}

class jit_constants {
    std::vector<std::shared_ptr<jit_constant>> _constants;
public:
    jit_constants(std::initializer_list<std::shared_ptr<jit_constant>> constants) :_constants(constants) {}

    void add_constant(std::shared_ptr<jit_constant> constant)
    {
        _constants.push_back(constant);
    }

    kernels_cache::jit_definitions get_definitions() const {
        kernels_cache::jit_definitions definitons;
        definitons.reserve(_constants.size() * 6); //assuming max 6 pairs per jit_constant

        for (auto& constant : _constants) {
            auto def = constant->get_definitions();
            definitons.insert(definitons.end(), def.begin(), def.end());
        }
        return definitons;
    }
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

class kernel_execution_options {
    const size_t lws_max = 256;
    std::vector<size_t> _global;
    std::vector<size_t> _local;

    void set_local_sizes();

    static cl::NDRange to_nd_range(const std::vector<size_t>& v)
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
            throw std::logic_error("Unacceptable NDRange dimension: " + std::to_string(v.size()));
        }
    }

public:
    kernel_execution_options(const std::vector<size_t>& work_items, const std::vector<size_t>& parallel_items) : _global(work_items), _local(parallel_items)
    {
        set_local_sizes();
        assert(_global.size() < 4 && _global.size() > 0 && _global.size() == _local.size());
    }

    kernel_execution_options(size_t work_items)
        : kernel_execution_options(std::vector<size_t>({ work_items }), std::vector<size_t>())
    {}

    kernel_execution_options(size_t work_items, size_t parallel_items)
        : kernel_execution_options(std::vector<size_t>({ work_items }), std::vector<size_t>({parallel_items}))
    {}

    kernel_execution_options(const std::vector<size_t>& work_items)
        : kernel_execution_options(work_items, std::vector<size_t>())
    {}

    kernel_execution_options(const std::vector<size_t>& work_items, size_t parallel_items)
        : kernel_execution_options(work_items, std::vector<size_t>({ parallel_items }))
    {}

    kernel_execution_options(const kernel_execution_options& other)
        : _global(other._global),
        _local(other._local) {}

    kernel_execution_options& operator=(const kernel_execution_options& other) {
        if (this == &other)
            return *this;
        _global = other._global;
        _local = other._local;
        return *this;
    }

    cl::NDRange global_range() const { return to_nd_range(_global); }
    cl::NDRange local_range() const { return to_nd_range(_local); }
};

class kernel : public context_holder {
    kernels_cache::kernel_id _kernel_id;

    template<unsigned index, typename Ti, typename... Ts>
    void setArgs(cl::Kernel& clkernel, Ti&& arg, Ts&&... args) const {
        clkernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
        setArgs<index + 1, Ts...>(clkernel, std::forward<Ts>(args)...);
    }


    template<unsigned index, typename Ti>
    void setArgs(cl::Kernel& clkernel, Ti&& arg) const {
        clkernel.setArg(index, kernel_arg_handler<Ti>::get(arg));
    }

    template<unsigned index>
    void setArgs(cl::Kernel&) const {}

public:
    explicit kernel(std::shared_ptr<gpu_toolkit> context, const std::string& template_id, kernels_cache::jit_definitions definitions = kernels_cache::jit_definitions(), const std::string& kernel_name = std::string())
        : context_holder(context), _kernel_id(context->get_kernels_cache().create_kernel_from_template(template_id, definitions, kernel_name)) {}
    explicit kernel(std::shared_ptr<gpu_toolkit> context, const std::string& template_id, const jit_constants& constants, const std::string& kernel_name = std::string())
        : context_holder(context), _kernel_id(context->get_kernels_cache().create_kernel_from_template(template_id, constants.get_definitions(), kernel_name)) {}

    kernel(const kernel& other) : context_holder(other.context()), _kernel_id(other._kernel_id) {}

    kernel& operator=(const kernel& other) {
        if (this == &other)
            return *this;
        _kernel_id = other._kernel_id;
        return *this;
    }

    template<typename... Args>
    cldnn::refcounted_obj_ptr<cldnn::event_impl> run(
        const kernel_execution_options& options,
        const std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>>& dependencies,
        Args... args) const
    {
        cl::Event end_event;
        std::vector<cl::Event> events;
        events.reserve(dependencies.size());
        for(auto& dependency : dependencies)
        {
            events.emplace_back(dependency->get());
        }

        auto clkernel = context()->get_kernels_cache().get_kernel(_kernel_id);
        setArgs<0>(clkernel, std::forward<Args>(args)...);
        context()->queue().enqueueNDRangeKernel(clkernel, cl::NullRange, options.global_range(), options.local_range(), &events, &end_event);

        return{ new cldnn::event_impl(end_event), false };
    }
};

} }
