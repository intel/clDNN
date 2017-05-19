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
#include "kernels_cache.h"
#include "ocl_toolkit.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>

namespace neural { namespace gpu {

const char program_dump_file_name[] = "clDNN_program.cl";

static const char* kernels_header = R"__krnl(
#define CAT(x, y) x##y
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))

#define TRANSPOSE_BLOCK_8( _block )   \
        (float8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_COL( _block, _col )   \
        (float8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_16_FP16(_block)                                  \
        (half16)(as_half2(intel_sub_group_shuffle(_block, 0)),  \
                 as_half2(intel_sub_group_shuffle(_block, 1)),  \
                 as_half2(intel_sub_group_shuffle(_block, 2)),  \
                 as_half2(intel_sub_group_shuffle(_block, 3)),  \
                 as_half2(intel_sub_group_shuffle(_block, 4)),  \
                 as_half2(intel_sub_group_shuffle(_block, 5)),  \
                 as_half2(intel_sub_group_shuffle(_block, 6)),  \
                 as_half2(intel_sub_group_shuffle(_block, 7)));

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}

#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

#define ADD_BIAS_16_FP16( _result, _biasVal) \
{ \
    _result.s01 += as_half2(intel_sub_group_shuffle(_biasVal, 0)); \
    _result.s23 += as_half2(intel_sub_group_shuffle(_biasVal, 1)); \
    _result.s45 += as_half2(intel_sub_group_shuffle(_biasVal, 2)); \
    _result.s67 += as_half2(intel_sub_group_shuffle(_biasVal, 3)); \
    _result.s89 += as_half2(intel_sub_group_shuffle(_biasVal, 4)); \
    _result.sab += as_half2(intel_sub_group_shuffle(_biasVal, 5)); \
    _result.scd += as_half2(intel_sub_group_shuffle(_biasVal, 6)); \
    _result.sef += as_half2(intel_sub_group_shuffle(_biasVal, 7)); \
}

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + byte_offset))

#define MULTIPLY_OFFSET(elem_type, byte_offset) (byte_offset * sizeof(elem_type))

)__krnl";

std::vector<std::string> kernels_cache::get_program_source() const {
    std::vector<std::string> source{ kernels_header };
    for (auto& code : _kernel_codes) {
        source.push_back(code.second);
    }
    return source;
}

namespace {

    class code_builder
    {
        std::ostringstream oss;
        std::string code;
        std::vector<std::string> defined_macroses;

        code_builder& register_macro(const std::string& name)
        {
            assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) == 0);
            defined_macroses.push_back(name);
            return *this;
        }

    public:
        code_builder& set_code(const std::string& c)
        {
            assert(code.empty());
            code = c;
            return *this;
        }

        code_builder& add_line(const std::string& line) {
            oss << line << "\n";
            return *this;
        }

        code_builder& decoration_macro(const std::string& name, const std::string& prefix, const std::string& postfix, const std::string& name_prefix = std::string())
        {
            oss << "#define " << name << "(name) " << prefix << " " + name_prefix + "_##" + "name" << (postfix.empty() ? "" : "##_") << postfix << std::endl;
            return register_macro(name);
        }


        code_builder& value_macro(const std::string& name, const std::string& value)
        {
            oss << "#define " << name << " " << value << std::endl;
            return register_macro(name.substr(0, name.find('(')));
        }

        std::string str()
        {
            std::ostringstream os;
            os << oss.str();
            os << code << std::endl;
            std::for_each(std::crbegin(defined_macroses), std::crend(defined_macroses), [&](const std::string& name) { os << "#undef " << name << std::endl; });
            return os.str();
        }
    };

}

kernels_cache::kernels_cache(gpu_toolkit& context): _context(context) {}

kernels_cache::kernel_id kernels_cache::create_kernel_from_template(const std::string& template_name, jit_definitions definitions, std::string kernel_name) {
    // TODO: FIXIT: more than one kernel can be created for same template_name and definitions

    std::string primitive_name = kernel_name;
    std::replace(kernel_name.begin(), kernel_name.end(), '.', '_');
    auto kernel_num = definitions.empty() ? "" : std::to_string(_kernel_codes.size());

    if (kernel_name.empty() || !_context.get_configuration().meaningful_kernels_names)
        kernel_name = template_name;

    kernel_name += (kernel_num.empty() ? "" : "_") + kernel_num;
    
    class code_builder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + template_name + " ")
        .add_line("// Kernel name: " + kernel_name)
        .add_line("// Primitive id: " + primitive_name)
        .value_macro("KERNEL(name)", "__kernel void " + kernel_name)
        .decoration_macro("FUNC", "", kernel_num)
        .decoration_macro("FUNC_CALL", "", kernel_num);
    for (auto& definition : definitions) {
        code.value_macro(definition.first, definition.second);
    }
    code.set_code(_database.get(template_name).at(0));

    auto kernel_code = code.str();

    std::lock_guard<std::mutex> lock(_mutex);
    _kernel_codes[kernel_name] = kernel_code;
    _modified = true;
    return kernel_name;
}

void kernels_cache::build_program() {
    try {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_modified) {
            auto program_source = get_program_source();
#ifndef NDEBUG
            {
                std::ofstream os(program_dump_file_name);
                for (auto& s : program_source)
                    os << s;
            }
#endif
            cl::Program program(_context.context(), program_source);
            program.build({ _context.device() }, "-cl-mad-enable");
#ifndef NDEBUG
            {
                std::ofstream os(program_dump_file_name, std::ios_base::app);
                os << "\n/* Build Log:\n";
                for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
                    os << p.second << "\n";
                }
                os << "*/\n";
            }
#endif
            cl::vector<cl::Kernel> kernels;
            program.createKernels(&kernels);
            _kernels.clear();
            for(auto& k : kernels)
            {
                auto kernel_name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                _kernels.emplace(kernel_name, k);
            }
        }
        _modified = false;
    }
    catch (const cl::BuildError& err) {
        std::string build_log{"Build program error "};
        build_log += err.what();
#ifndef NDEBUG
        {
            std::ofstream os(program_dump_file_name, std::ios_base::app);
            os << "\n/* Build Log:\n";
            for (auto& p : err.getBuildLog()) {
                os << p.second << "\n";
                build_log += "\n" + p.second;
            }
            os << "*/\n";
        }
#endif
        throw std::runtime_error(build_log);
    }
}

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id) {
    build_program();
    return _kernels.at(id);
}

}}
