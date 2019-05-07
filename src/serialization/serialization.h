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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/forward_list.hpp>
#include <boost/serialization/wrapper.hpp>
#include <boost/serialization/assume_abstract.hpp>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calling serialization methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    IF_INCLUDED
*/
/// @brief Call simple expression.
#define CLDNN_SERIALIZATION_IF_INCLUDED(Expression)                                                                             \
    Expression


/*
    NVP
*/
/// @brief Call for BOOST_SERIALIZATION_NVP
#define CLDNN_SERIALIZATION_NVP(Name)                                                                                           \
    BOOST_SERIALIZATION_NVP(Name)


/*
    BASE_OBJECT_NVP
*/
/// @brief Call for BOOST_SERIALIZATION_BASE_OBJECT_NVP
#define CLDNN_SERIALIZATION_BASE_OBJECT_NVP(Name)                                                                               \
    BOOST_SERIALIZATION_BASE_OBJECT_NVP(Name)


/*
    BASE_OBJECT_NVP_PRIMITIVE_BASE
*/
/// @brief Serialize Base object @class primitive_base.
#define CLDNN_SERIALIZATION_BASE_OBJECT_NVP_PRIMITIVE_BASE(PType)                                                               \
    boost::serialization::make_nvp("primitive_base",                                                                            \
        boost::serialization::base_object<                                                                                      \
        primitive_base<PType, CLDNN_PRIMITIVE_DESC(PType)>                                                                      \
        >(*this))


/*
    MEMBERS
*/
/// @brief Define serialize template method. Call it in class definition, in private section.
/// @details Example:
    /*! @code
    *
    class A : public B
    {
    public:
        int member;
    private:
        CLDNN_SERIALIZATION_MEMBERS(
        ar & CLDNN_SERIALIZATION_BASE_OBJECT_NVP(B)
           & CLDNN_SERIALIZATION_NVP(first member) & CLDNN_SERIALIZATION_NVP(second member)...;
        )
    }
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_MEMBERS(Members)                                                                                    \
    friend class boost::serialization::access;                                                                                  \
    template<class Archive>                                                                                                     \
    void serialize(Archive &ar, const unsigned int /*version*/)                                                                 \
    {                                                                                                                           \
        Members                                                                                                                 \
    }


/*
    PARENT_ONLY
*/
/// @brief Define template method which serialize only class parent. 
/// Make sure that class has parent definition.
/// @details Example:
    /*! @code
    *
    class A : public B
    {
    public:
        using parent = B;
    private:
        CLDNN_SERIALIZATION_PARENT_ONLY()
    }
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_PARENT_ONLY()                                                                                       \
    CLDNN_SERIALIZATION_MEMBERS                                                                                                 \
    (                                                                                                                           \
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(parent);                                                                       \
    )


/*
    PROGRAM_SERIALIZE
*/
/// @brief Define program_impl serialize method.
#define CLDNN_SERIALIZATION_PROGRAM_SERIALIZE()                                                                                 \
    friend class boost::serialization::access;                                                                                  \
    template<class Archive>                                                                                                     \
    void serialize(Archive &ar, const unsigned int /*version*/)                                                                 \
    {                                                                                                                           \
        const auto cldnn_version = cldnn::get_version();                                                                        \
        const auto driver_version = engine->get_engine_info().driver_version;                                                   \
        const uint32_t hashed_cldnn_version_current =                                                                           \
            (cldnn_version.major << 28) ^ (cldnn_version.minor << 24) ^ (cldnn_version.build << 14) ^ cldnn_version.revision;   \
                                                                                                                                \
        uint32_t hashed_cldnn_version_saved;                                                                                    \
        std::string driver_version_saved;                                                                                       \
        if (Archive::is_saving::value)                                                                                          \
        {                                                                                                                       \
            hashed_cldnn_version_saved = hashed_cldnn_version_current;                                                          \
            driver_version_saved = driver_version;                                                                              \
        }                                                                                                                       \
        ar & CLDNN_SERIALIZATION_NVP(hashed_cldnn_version_saved) & CLDNN_SERIALIZATION_NVP(driver_version_saved);               \
                                                                                                                                \
        if (hashed_cldnn_version_current == hashed_cldnn_version_saved && driver_version_saved == driver_version)               \
        {                                                                                                                       \
            ar & CLDNN_SERIALIZATION_NVP(prog_id) & CLDNN_SERIALIZATION_NVP(processing_order) & CLDNN_SERIALIZATION_NVP(inputs) \
               & CLDNN_SERIALIZATION_NVP(outputs) & CLDNN_SERIALIZATION_NVP(nodes_map) & CLDNN_SERIALIZATION_NVP(optimized_out);\
        }                                                                                                                       \
        else                                                                                                                    \
            throw std::runtime_error("clDNN or driver versions are different!");                                                \
    }



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Save and load
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    PROGRAM_SAVE
*/
/// @brief Call save_program method.
#define CLDNN_SERIALIZATION_PROGRAM_SAVE(Program, BuildOption, IsInternal)                                                      \
    auto name = BuildOption.get<build_option_type::serialize_program>()->serialization_program_name;                            \
    if (!name.empty() && !IsInternal)                                                                                           \
    {                                                                                                                           \
        Program->save_program(name);                                                                                            \
    }


/*
    PROGRAM_LOAD
*/
/// @brief Call load_program method and compile cl_program from binaries.
#define CLDNN_SERIALIZATION_PROGRAM_LOAD(Engine, Name, DumpPath)                                                                \
    auto prog_impl = new program_impl(*Engine);                                                                                 \
    prog_impl->load_program(Name, DumpPath);                                                                                    \
    return{ prog_impl, false };


/*
    SAVE
*/
/// @brief Save program.
#define CLDNN_SERIALIZATION_SAVE(Program, Name)                                                                                 \
    std::ofstream fstream(Name + ".xml");                                                                                       \
    assert(fstream.good());                                                                                                     \
    boost::archive::xml_oarchive xml_archive(fstream);                                                                          \
    xml_archive & boost::serialization::make_nvp("program", *Program);                                                          \
                                                                                                                                \
    std::ofstream binary_ofstream(Name + "_binaries.bin", std::ios::binary);                                                    \
    assert(binary_ofstream.good());                                                                                             \
    boost::archive::binary_oarchive boa(binary_ofstream);                                                                       \
    boa & *Program->get_engine().get_context()->get_binaries();                                                                 \
    serialize_binary(boa, processing_order);


/*
    LOAD
*/
/// @brief Load program.
#define CLDNN_SERIALIZATION_LOAD(Program, Name, DumpPath)                                                                       \
    std::ifstream fstream(Name + ".xml");                                                                                       \
    assert(fstream.good());                                                                                                     \
    boost::archive::xml_iarchive xml_archive(fstream);                                                                          \
    xml_archive & boost::serialization::make_nvp("program", *Program);                                                          \
    Program->options.set_option(cldnn::build_option::graph_dumps_dir(DumpPath));;                                               \
                                                                                                                                \
    std::ifstream binary_ifstream(Name + "_binaries.bin", std::ios::binary);                                                    \
    assert(binary_ifstream.good());                                                                                             \
    boost::archive::binary_iarchive bia(binary_ifstream);                                                                       \
    bia & *Program->get_engine().get_context()->get_binaries();                                                                 \
    serialize_binary(bia, processing_order);



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Overload constructors
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    OVERLOAD_NT_GPU_CLASS_CONSTRUCTOR
*/
///@brief Overloads constructor of non typical (NT) @param PType gpu class. NT -> one argument(PTyped_node).
#define CLDNN_SERIALIZATION_OVERLOAD_NT_GPU_CLASS_CONSTRUCTOR(Class, PType)                                                     \
namespace boost {                                                                                                               \
    namespace serialization {                                                                                                   \
        template<class Archive>                                                                                                 \
        void save_construct_data(Archive & ar, const Class * t, const unsigned int /*file_version*/)                            \
        {                                                                                                                       \
            auto outer = &t->outer;                                                                                             \
            ar << make_nvp("outer", outer);                                                                                     \
        }                                                                                                                       \
        template<class Archive>                                                                                                 \
        void load_construct_data(Archive & ar, Class * t, const unsigned int /*file_version*/)                                  \
        {                                                                                                                       \
            cldnn::typed_program_node<cldnn::PType>* outer;                                                                     \
            ar >> make_nvp("outer", outer);                                                                                     \
            ::new(static_cast<void*>(t))Class(*outer);                                                                          \
        }                                                                                                                       \
    }                                                                                                                           \
}


/*
    OVERLOAD_INTERNAL_NODE_CONSTRUCTOR
*/
///@brief Overloads internal_typed_program_node_base constructor.
#define CLDNN_SERIALIZATION_OVERLOAD_INTERNAL_NODE_CONSTRUCTOR(InternalPType)                                                   \
namespace boost {                                                                                                               \
    namespace serialization {                                                                                                   \
        template<class Archive>                                                                                                 \
        void save_construct_data(Archive & ar, const InternalPType * t, const unsigned int /*file_version*/)                    \
        {                                                                                                                       \
            auto myprog = &t->get_program();                                                                                    \
            ar << make_nvp("myprog", myprog);                                                                                   \
        }                                                                                                                       \
        template<class Archive>                                                                                                 \
        void load_construct_data(Archive & ar, InternalPType * t, const unsigned int /*file_version*/)                          \
        {                                                                                                                       \
            cldnn::program_impl* myprog;                                                                                        \
            ar >> make_nvp("myprog", myprog);                                                                                   \
            ::new(static_cast<void*>(t))InternalPType(*myprog);                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
}


/*
    OVERLOAD_API_NODE_CONSTRUCTOR
*/
///@brief Overloads api_typed_program_node_base constructor.
#define CLDNN_SERIALIZATION_OVERLOAD_API_NODE_CONSTRUCTOR(Class, PType)                                                         \
namespace boost {                                                                                                               \
    namespace serialization {                                                                                                   \
        template<class Archive>                                                                                                 \
        void save_construct_data(Archive & ar, const Class * t, const unsigned int /*file_version*/)                            \
        {                                                                                                                       \
            auto desc = t->get_primitive();                                                                                     \
            auto myprog = &t->get_program();                                                                                    \
            ar << make_nvp("desc", desc)                                                                                        \
               << make_nvp("myprog", myprog);                                                                                   \
        }                                                                                                                       \
        template<class Archive>                                                                                                 \
        void load_construct_data(Archive & ar, Class * t, const unsigned int /*file_version*/)                                  \
        {                                                                                                                       \
            std::shared_ptr<cldnn::PType> desc;                                                                                 \
            cldnn::program_impl* myprog;                                                                                        \
            ar >> make_nvp("desc", desc)                                                                                        \
               >> make_nvp("myprog", myprog);                                                                                   \
            ::new(static_cast<void*>(t))Class(desc, *myprog);                                                                   \
        }                                                                                                                       \
    }                                                                                                                           \
}


/*
    OVERLOAD_GPU_CLASS_CONSTRUCTOR
*/
///@brief Overloads PType_gpu constructor.
#define CLDNN_SERIALIZATION_OVERLOAD_GPU_CLASS_CONSTRUCTOR(Class, PType)                                                        \
namespace boost {                                                                                                               \
    namespace serialization {                                                                                                   \
        template<class Archive>                                                                                                 \
        void save_construct_data(Archive & ar, const Class * t, const unsigned int /*file_version*/)                            \
        {                                                                                                                       \
            auto outer = &t->_outer;                                                                                            \
            ar << make_nvp("_outer", outer)                                                                                     \
               << make_nvp("_kernels_data", t->_kernel_data);                                                                   \
        }                                                                                                                       \
        template<class Archive>                                                                                                 \
        void load_construct_data(Archive & ar, Class * t, const unsigned int /*file_version*/)                                  \
        {                                                                                                                       \
            cldnn::typed_program_node<cldnn::PType>* outer;                                                                     \
            kernel_selector::kernel_data kernel_data;                                                                           \
            ar >> make_nvp("_outer", outer)                                                                                     \
               >> make_nvp("_kernels_data", kernel_data);                                                                       \
            ::new(static_cast<void*>(t))Class(*outer, kernel_data);                                                             \
        }                                                                                                                       \
    }                                                                                                                           \
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Keys and implements export
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    EXPORT_NODE_KEY
*/
/// @brief Export keys of @param PType primitive class. Call it at the end of PType.hpp file.
/// @details Example:
    // PType.hpp file
    /*! @code
    *
    namespace cldnn
    {
        ...
    }
    CLDNN_SERIALIZATION_EXPORT_NODE_KEY(PType)
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_EXPORT_NODE_KEY(PType)                                                                              \
    BOOST_CLASS_EXPORT_KEY2(cldnn::PType, #PType)


/*
    EXPORT_NODE_IMPLEMENT
*/
/// @brief Export implements of non typical class (typical uses macro which contains several calls of BOOS_CLASS_EXPORT_IMPLEMENT)
/// For example internal_program_node_base class. Call it at the end of program_node.cpp
    // program_node.cpp file
    /*! @code
    *
    ...
    CLDNN_SERIALIZATION_EXPORT_NODE_IMPLEMENT(full_namespace::internal_program_node_base)
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_EXPORT_NODE_IMPLEMENT(ExtPType)                                                                     \
    BOOST_CLASS_EXPORT_IMPLEMENT(ExtPType)


/*
    EXPORT_NODE_IMPLEMENTS
*/
/// @brief Export implements of typical @param PType classes derived from progam_node. Call it at the end of PType.cpp
/// @details Example:
    // PType.cpp file
    /*! @code
    *
    ...
    CLDNN_SERIALIZATION_EXPORT_NODE_IMPLEMENTS(PType)
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_EXPORT_NODE_IMPLEMENTS(PType)                                                                       \
    BOOST_CLASS_EXPORT_IMPLEMENT(cldnn::PType)                                                                                  \
    BOOST_CLASS_EXPORT_IMPLEMENT(cldnn::typed_program_node_base<cldnn::PType>)                                                  \
    BOOST_CLASS_EXPORT_IMPLEMENT(cldnn::typed_program_node<cldnn::PType>)


/*
    EXPORT_CLASS_GUID
*/
/// @brief Export GUIDs of non typical class (typical uses macro which contains several calls of BOOST_CLASS_EXPORT_GUID)
/// @details Example: wait_for_events_gpu. Call it at the end of wait_for_events_gpu.cpp
    // program_node.cpp file
    /*! @code
    *
    namespace cldnn
    {
        namespace gpu
        {
            ...
        }
    }
    CLDNN_SERIALIZATION_EXPORT_CLASS_GUID(cldnn::gpu::wait_for_events_gpu, "wait_for_events_gpu")
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_EXPORT_CLASS_GUID(Class, Name)                                                                      \
    BOOST_CLASS_EXPORT_KEY2(Class, Name)                                                                                        \
    BOOST_CLASS_EXPORT_IMPLEMENT(Class)



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calling multiple macros
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    PROGRAM_NODE
*/
/// @brief Assumes abstract for three basic classes, overloads program_node and internal_program_node_base constructors,
/// exports internal key. Call it at the end of program_node.h file. 
#define CLDNN_SERIALIZATION_PROGRAM_NODE()                                                                                      \
    BOOST_SERIALIZATION_ASSUME_ABSTRACT(primitive)                                                                              \
    BOOST_SERIALIZATION_ASSUME_ABSTRACT(primitive_impl)                                                                         \
    BOOST_SERIALIZATION_ASSUME_ABSTRACT(program_node)                                                                           \
    BOOST_CLASS_EXPORT_KEY2(cldnn::details::internal_program_node_base, "internal_program_node_base")                           \
    CLDNN_SERIALIZATION_OVERLOAD_API_NODE_CONSTRUCTOR(cldnn::program_node, primitive)                                           \
    CLDNN_SERIALIZATION_OVERLOAD_INTERNAL_NODE_CONSTRUCTOR(cldnn::details::internal_program_node_base)


/*
    TYPED_PROGRAM_NODE_CLASS
*/
/// @brief Exports key and overloads constructors of @param PType base classes. Call it at the end of PType_inst.h file. 
/// @details Example:
    // PType_inst.h file
    /*! @code
    *
    namespace cldnn
    {
        ...
    }
    CLDNN_SERIALIZATION_TYPED_PROGRAM_NODE_CLASS(PType)
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_TYPED_PROGRAM_NODE_CLASS(PType)                                                                     \
    BOOST_CLASS_EXPORT_KEY2(cldnn::typed_program_node_base<cldnn::PType>, BOOST_PP_STRINGIZE(PType##_node_base))                \
    BOOST_CLASS_EXPORT_KEY2(cldnn::typed_program_node<cldnn::PType>, BOOST_PP_STRINGIZE(PType##_node))                          \
    CLDNN_SERIALIZATION_OVERLOAD_API_NODE_CONSTRUCTOR(cldnn::details::api_typed_program_node_base<cldnn::PType>, PType)         \
    CLDNN_SERIALIZATION_OVERLOAD_API_NODE_CONSTRUCTOR(cldnn::typed_program_node<cldnn::PType>, PType)


/*
    GPU_NG_CLASS
*/
/// @brief Exports primitive GUIDs and overloads default constructors of @param PType classes, derived 
/// from typed_primitive_impl (not typed_primitive_gpu_impl). Call it at the end of PType_gpu.cpp file. 
/// @details Example:
    // PType_gpu.cpp file
    /*! @code
    *
    namespace cldnn
    {
    namespace gpu
        {
            ...
        }
    }
    CLDNN_SERIALIZATION_GPU_NG_CLASS(PType)
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_GPU_NG_CLASS(Namespace, PType)                                                                      \
    CLDNN_SERIALIZATION_EXPORT_CLASS_GUID(cldnn::typed_primitive_impl<cldnn::PType>, "typed_primitive_impl")                    \
    CLDNN_SERIALIZATION_EXPORT_CLASS_GUID(Namespace::PType##_gpu, BOOST_PP_STRINGIZE(PType##_gpu))                              \
    CLDNN_SERIALIZATION_OVERLOAD_NT_GPU_CLASS_CONSTRUCTOR(Namespace::PType##_gpu, PType)


/*
    GPU_CLASS
*/
/// @brief Exports primitive GUIDs and overloads default constructors of @param PType classes, derived 
/// from typed_primitive_gpu_impl. Call it at the end of PType_gpu.cpp file. 
/// @details Example:
    // PType_gpu.cpp file
    /*! @code
    *
    namespace cldnn
    {
        namespace gpu
        {
            ...
        }
    }
    CLDNN_SERIALIZATION_GPU_CLASS(PType)
    *
    *@endcode
    */
#define CLDNN_SERIALIZATION_GPU_CLASS(PType)                                                                                    \
    BOOST_CLASS_EXPORT_GUID(cldnn::typed_primitive_impl<cldnn::PType>, "typed_primitive_impl")                                  \
    BOOST_CLASS_EXPORT_GUID(cldnn::gpu::PType##_gpu::parent, BOOST_PP_STRINGIZE(PType##_gpu_impl))                              \
    BOOST_CLASS_EXPORT_GUID(cldnn::gpu::PType##_gpu, BOOST_PP_STRINGIZE(PType##_gpu))                                           \
    CLDNN_SERIALIZATION_OVERLOAD_GPU_CLASS_CONSTRUCTOR(cldnn::gpu::PType##_gpu::parent, PType)                                  \
    CLDNN_SERIALIZATION_OVERLOAD_GPU_CLASS_CONSTRUCTOR(cldnn::gpu::PType##_gpu, PType)
