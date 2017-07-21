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
#include <set>

#include "kernel_selector_helper.h"

#ifndef NDEBUG
#define OUT_PORGRAM_TO_FILE
#endif

namespace neural { namespace gpu {

const char program_dump_file_name[] = "clDNN_program";

namespace 
{
    std::string get_undef_jit(kernels_cache::source_code org_source_code)
    {
        const std::string white_space_with_new_lines = " \t\r\n";
        const std::string white_space = " \t";

        size_t current_pos = 0;

        const std::string define = "define";

        std::set<std::string> to_undef;
        for (const auto& source : org_source_code)
        {
            do
            {
                size_t index_to_hash = source.find_first_not_of(white_space_with_new_lines, current_pos);
                if (index_to_hash != std::string::npos &&
                    source[index_to_hash] == '#')
                {
                    size_t index_define = source.find_first_not_of(white_space, index_to_hash + 1);

                    if (index_define != std::string::npos &&
                        !source.compare(index_define, define.size(), define))
                    {
                        size_t index_to_name = source.find_first_not_of(white_space, index_define + define.size());
                        if (index_to_name != std::string::npos)
                        {
                            size_t index_to_end_name = source.find_first_of(white_space_with_new_lines + "(", index_to_name);
                            if (index_to_end_name == std::string::npos)
                            {
                                index_to_end_name = source.size();
                            }
                            std::string name = source.substr(index_to_name, index_to_end_name - index_to_name);
                            to_undef.insert(name);
                        }
                    }
                }

                current_pos = source.find_first_of('\n', current_pos + 1);
            } while (current_pos != std::string::npos);
        }

        std::string undefs;
        for (const auto& name : to_undef)
        {
            undefs += "#ifdef " + name + "\n";
            undefs += "#undef " + name + "\n";
            undefs += "#endif\n";
        }

        return std::move(undefs);
    }

    std::string reroder_options(const std::string& org_options)
    {
        std::stringstream ss(org_options);
        std::set<std::string> sorted_options;

        while (ss.good())
        {
            std::string word;
            ss >> word;
            sorted_options.insert(word);
        }

        std::string options;

        for (const auto& o : sorted_options)
        {
            options += o + " ";
        }
        
        return options;
    }

    inline bool does_options_support_batch_compilation(const std::string& options)
    {
        return
            options.find("-D") == std::string::npos &&
            options.find("-I") == std::string::npos;
    }
}

kernels_cache::sorted_code kernels_cache::get_program_source(const kernels_code& kernels_source_code) const 
{
    sorted_code scode;

    for (const auto& code : kernels_source_code)
    {
        const source_code   org_source_code     = { code.second.kernel_strings->jit, code.second.kernel_strings->str };
        std::string         entry_point         = code.second.kernel_strings->entry_point;
        std::string         options             = code.second.kernel_strings->options;
        bool                batch_compilation   = code.second.kernel_strings->batch_compilation;
        bool                dump_custom_program = code.second.dump_custom_program;

        batch_compilation &= does_options_support_batch_compilation(options);

        if (batch_compilation)
        {
            options = reroder_options(options);
        }

        std::string key = options;

        if (batch_compilation == false)
        {
            key += " __PROGRAM__" + std::to_string(scode.size());
        }

        if (dump_custom_program)
        {
            key += " __DUMP_CUSTOM_PROGRAM__"; // Adding label to key so it would be separated from other programs
        }

        auto& current_bucket = scode[key];
        current_bucket.dump_custom_program = dump_custom_program;

        if (current_bucket.source.empty())
        {
            current_bucket.options = options;
        }

        current_bucket.entry_point_to_id[entry_point] = code.second.id;

        source_code new_source_code = org_source_code;

        if (batch_compilation)
        {
            new_source_code.push_back(get_undef_jit(org_source_code));
        }

        for (auto& s : new_source_code)
        {
            current_bucket.source.push_back(std::move(s));
        }
    }

    return std::move(scode);
}

kernels_cache::kernels_cache(gpu_toolkit& context): _context(context) {}

kernels_cache::kernel_id kernels_cache::set_kernel_source(const std::shared_ptr<kernel_selector::kernel_string>& kernel_string, bool dump_custom_program)
{
    kernels_cache::kernel_id id;
    
    // same kernel_string == same kernel
    const auto key = kernel_string.get();

    std::lock_guard<std::mutex> lock(_mutex);

    const auto it = _kernels_code.find(key);

    if (it == _kernels_code.end())
    {
        // we need unique id in order to avoid conflict across topologies.
        const auto kernel_num = _kernels.size() + _kernels_code.size(); 
        id = kernel_string->entry_point + "_" + std::to_string(kernel_num);
        _kernels_code[key] = { kernel_string, id, dump_custom_program };
    }
    else
    {
        id = it->second.id;
    }

    assert(_kernels.find(id) == _kernels.end());

    return id;
}

kernels_cache::kernels_map kernels_cache::build_program(const program_code& program_source) const
{
    static uint32_t current_file_index = 0;
    const std::string current_program_dump_file_name = program_dump_file_name + std::to_string(current_file_index) + ".cl";

    try 
    {
#ifndef OUT_PORGRAM_TO_FILE
        if (program_source.dump_custom_program)
#endif
        {
            current_file_index++;
            std::ofstream os(current_program_dump_file_name);
            for (auto& s : program_source.source)
                os << s;
        }

        cl::Program program(_context.context(), program_source.source);
        program.build({ _context.device() }, program_source.options.c_str());

#ifndef OUT_PORGRAM_TO_FILE
        if (program_source.dump_custom_program)
#endif
        {
            std::ofstream os(current_program_dump_file_name, std::ios_base::app);
            os << "\n/* Build Log:\n";
            for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
                os << p.second << "\n";
            }
            os << "*/\n";
        }

        cl::vector<cl::Kernel> kernels;
        program.createKernels(&kernels);
        kernels_map kmap;

        for (auto& k : kernels)
        {
            auto kernel_name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
            kmap.emplace(kernel_name, k);
        }

        return std::move(kmap);
    }
    catch (const cl::BuildError& err) 
    {
        std::string build_log{"Build program error "};
        build_log += err.what();

#ifndef OUT_PORGRAM_TO_FILE
        if (program_source.dump_custom_program)
#endif
        {
            std::ofstream os(current_program_dump_file_name, std::ios_base::app);
            os << "\n/* Build Log:\n";
            for (auto& p : err.getBuildLog()) {
                os << p.second << "\n";
                build_log += "\n" + p.second;
            }
            os << "*/\n";
        }

        throw std::runtime_error(build_log);
    }
}

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id) 
{
    std::lock_guard<std::mutex> lock(_mutex);
    
    if (_kernels_code.empty() == false) 
    {
        auto sorted_program_code = get_program_source(_kernels_code);

        for (auto& program : sorted_program_code)
        {
            auto kernels = build_program(program.second);

            for (auto& k : kernels)
            {
                const auto& entry_point = k.first;
                const auto& k_id = program.second.entry_point_to_id[entry_point];
                _kernels[k_id] = k.second;
            }
        }

        _kernels_code.clear();
    }

    return _kernels.at(id);
}

}}
