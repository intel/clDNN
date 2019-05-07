/*
// Copyright (c) 2019 Intel Corporation
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CLDNN_SERIALIZATION
#pragma once
#include "src/serialization/serialization.h"
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief Condition if clDNN serialization feature is not included.
#ifndef CLDNN_SERIALIZATION
// Empty or throws error if clDNN serialization feature is not included.

/*
    Calling serialization methods
*/
/// @brief Call expression.
#define CLDNN_SERIALIZATION_IF_INCLUDED(Expression)
/// @brief Call for BOOST_SERIALIZATION_NVP
#define CLDNN_SERIALIZATION_NVP(Name)
/// @brief Call for BOOST_SERIALIZATION_BASE_OBJECT_NVP
#define CLDNN_SERIALIZATION_BASE_OBJECT_NVP(Name)
/// @brief Serialize Base object @class primitive_base.
#define CLDNN_SERIALIZATION_BASE_OBJECT_NVP_PRIMITIVE_BASE(PType)
/// @brief Define serialize template method. Call it in class definition, in private section.
#define CLDNN_SERIALIZATION_MEMBERS(Members)
/// @brief Define template method which serialize only class parent. 
/// Make sure that class has parent definition.
#define CLDNN_SERIALIZATION_PARENT_ONLY()
/// @brief Define program_impl serialize method.
#define CLDNN_SERIALIZATION_PROGRAM_SERIALIZE()

/*
    Save and load
*/
/// @brief Call save_program method.
#define CLDNN_SERIALIZATION_PROGRAM_SAVE(Program, BuildOption, IsInternal)                                                      \
    auto name = BuildOption.get<build_option_type::serialize_program>()->serialization_program_name;                            \
    if (!name.empty() && !IsInternal)                                                                                           \
    {                                                                                                                           \
        std::cout << "Could not saved " << BuildOption.get<build_option_type::serialize_program>()->serialization_program_name  \
                  <<  ", clDNN serialization feature not included!" << std::endl;                                               \
    }
/// @brief Throws error if constructor is called.
#define CLDNN_SERIALIZATION_PROGRAM_LOAD(Program, Name, DumpPath)                                                               \
    throw std::runtime_error(Name + " program could not be loaded, clDNN serialization feature not included!");                 \
    (void)DumpPath;
/// @brief Throws error if trying to save program.
#define CLDNN_SERIALIZATION_SAVE(Program, Name)                                                                                 \
    std::cout << Name << " program  could not be saved, clDNN serialization feature not included!" << std::endl;
/// @brief Throws error if trying to load program.
#define CLDNN_SERIALIZATION_LOAD(Program, Name, DumpPath)                                                                       \
    throw std::runtime_error(Name + " program  could not be loaded, clDNN serialization feature not included!");                \
    (void)DumpPath;

/*
    Overload constructors
*/
#define CLDNN_SERIALIZATION_OVERLOAD_NT_GPU_CLASS_CONSTRUCTOR(Class, PType) 
#define CLDNN_SERIALIZATION_OVERLOAD_INTERNAL_NODE_CONSTRUCTOR(InternalPType)
#define CLDNN_SERIALIZATION_OVERLOAD_API_NODE_CONSTRUCTOR(Class, PType)
#define CLDNN_SERIALIZATION_OVERLOAD_GPU_CLASS_CONSTRUCTOR(Class, PType)

/*
    Keys and implements export
*/
#define CLDNN_SERIALIZATION_EXPORT_NODE_KEY(PType)
#define CLDNN_SERIALIZATION_EXPORT_NODE_IMPLEMENT(ExtPType)
#define CLDNN_SERIALIZATION_EXPORT_NODE_IMPLEMENTS(PType)
#define CLDNN_SERIALIZATION_EXPORT_CLASS_GUID(ExtPType, Name)

/*
    Calling multiple macros
*/
#define CLDNN_SERIALIZATION_PROGRAM_NODE()
#define CLDNN_SERIALIZATION_TYPED_PROGRAM_NODE_CLASS(PType)
#define CLDNN_SERIALIZATION_GPU_NG_CLASS(Namespace, PType)
#define CLDNN_SERIALIZATION_GPU_CLASS(PType)
#endif
