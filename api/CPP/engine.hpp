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
#include "cldnn_defs.h"

namespace cldnn
{

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_engine Execution Engine
/// @{

/// @brief Defines available engine types
enum class engine_types : int32_t
{
    ocl = cldnn_engine_ocl
};

/// @brief Configuration parameters for created engine.
struct engine_configuration
{
    const bool enable_profiling;         ///< Enable per-primitive profiling.
    const bool meaningful_kernels_names; ///< Generate meaniful names fo OpenCL kernels.
    const bool dump_custom_program;      ///< Dump the user OpenCL programs to files
    const std::string compiler_options;  ///< OpenCL compiler options string.
    const std::string single_kernel_name; ///< If provided, runs specific layer.

    /// @brief Constructs engine configuration with specified options.
    /// @param profiling Enable per-primitive profiling.
    /// @param decorate_kernel_names Generate meaniful names fo OpenCL kernels.
    /// @param dump_custom_program Dump the custom OpenCL programs to files
    /// @param options OpenCL compiler options string.
    /// @param single_kernel If provided, runs specific layer.
    engine_configuration(bool profiling = false, bool decorate_kernel_names = false, bool dump_custom_program = false, const std::string& options = std::string(), const std::string& single_kernel = std::string())
        :enable_profiling(profiling), meaningful_kernels_names(decorate_kernel_names), dump_custom_program(dump_custom_program), compiler_options(options), single_kernel_name(single_kernel) {}

    engine_configuration(const cldnn_engine_configuration& c_conf)
        :enable_profiling(c_conf.enable_profiling != 0), meaningful_kernels_names(c_conf.meaningful_kernels_names != 0), dump_custom_program(c_conf.dump_custom_program != 0), compiler_options(c_conf.compiler_options), single_kernel_name(c_conf.single_kernel_name){}

    /// @brief Implicit conversion to C API @ref ::cldnn_engine_configuration
    operator ::cldnn_engine_configuration() const
    {
        return{ enable_profiling, meaningful_kernels_names, dump_custom_program, compiler_options.c_str(), single_kernel_name.c_str() };
    }
};

/// @brief Information about the engine properties and capabilities.
/// @details Look into @ref ::cldnn_engine_info for details.
using engine_info = ::cldnn_engine_info;

/// @brief Represents clDNN engine object.
struct engine
{
    /// @brief Constructs @p OpenCL engine
    engine(const engine_configuration& configuration = engine_configuration())
        :engine(engine_types::ocl, 0, configuration)
    {}

    /// @brief Construct engine of the specified @p type, @p engine_num, and @p configuration options.
    /// @param[in] type Engine type @ref cldnn_engine_type. Only OCL engine is supported.
    /// @param[in] engine_num Engine index. Should be 0.
    /// @param[in] configuration Pointer to engine configuration options.
    engine(engine_types type, uint32_t engine_num, const engine_configuration& configuration = engine_configuration())
        :_impl(check_status<::cldnn_engine>("failed to create engine", [&](status_t* status)
              {
                  cldnn_engine_configuration conf = configuration;
                  return cldnn_create_engine(static_cast<int32_t>(type), engine_num, &conf, status);
              }))
    {}

    // TODO add move construction/assignment
    engine(const engine& other) :_impl(other._impl)
    {
        retain();
    }

    engine& operator=(const engine& other)
    {
        if (_impl == other._impl) return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    ~engine()
    {
        release();
    }

    friend bool operator==(const engine& lhs, const engine& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const engine& lhs, const engine& rhs) { return !(lhs == rhs); }

    /// @brief Returns number of available engines of the particular @p type.
    static uint32_t engine_count(engine_types type)
    {
        return check_status<uint32_t>("engine_count failed", [=](status_t* status)
        {
            return cldnn_get_engine_count(static_cast<int32_t>(type), status);
        });
    }

    /// @brief Returns information about properties and capabilities for the engine.
    engine_info get_info() const
    {
        return check_status<engine_info>("engine_count failed", [=](status_t* status)
        {
            return cldnn_get_engine_info(_impl, status);
        });
    }

    /// @brief Returns type of the engine.
    engine_types get_type() const
    {
        return check_status<engine_types>("engine_count failed", [=](status_t* status)
        {
            return static_cast<engine_types>(cldnn_get_engine_type(_impl, status));
        });
    }

    /// @brief get C API engine handler.
    ::cldnn_engine get() const { return _impl; }

private:
    friend struct network;
    friend struct memory;
    friend struct event;
    engine(::cldnn_engine impl) : _impl(impl)
    {
        if (_impl == nullptr) throw std::invalid_argument("implementation pointer should not be null");
    }
    ::cldnn_engine _impl;

    void retain()
    {
        check_status<void>("retain engine failed", [=](status_t* status) { cldnn_retain_engine(_impl, status); });
    }
    void release()
    {
        check_status<void>("release engine failed", [=](status_t* status) { cldnn_release_engine(_impl, status); });
    }
};
CLDNN_API_CLASS(engine)

/// @}

/// @}

}
