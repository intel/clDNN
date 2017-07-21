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
#include <map>
#include <mutex>
#include <vector>
#include <memory>

namespace cl {
class Kernel;
}

namespace KernelSelector
{
    struct KernelString;
}

namespace kernel_selector
{
    using kernel_string = KernelSelector::KernelString;
}

namespace neural {namespace gpu {
class gpu_toolkit;

class kernels_cache {
public:
    using source_code = std::vector<std::string>;

    struct program_code
    {
        source_code source;
        std::string options;
        bool dump_custom_program;
        std::map<std::string, std::string> entry_point_to_id;
    };

    struct kernel_code
    {
        std::shared_ptr<kernel_selector::kernel_string> kernel_strings;
        std::string id;
        bool dump_custom_program;
    };

    typedef std::string kernel_id;
    typedef cl::Kernel kernel_type;
    using sorted_code = std::map<std::string, program_code>;
    using kernels_map = std::map<std::string, kernel_type>;
    using kernels_code = std::map<void*, kernel_code>;

private:
    gpu_toolkit& _context;
    std::mutex _mutex;
    kernels_code _kernels_code;
    std::map<std::string, kernel_type> _kernels;

    sorted_code get_program_source(const kernels_code& kernels_source_code) const;
    friend class gpu_toolkit;
    explicit kernels_cache(gpu_toolkit& context);
    kernels_map build_program(const program_code& pcode) const;

public:
    kernel_id set_kernel_source(const std::shared_ptr<kernel_selector::kernel_string>& kernel_string, bool dump_custom_program);
    kernel_type get_kernel(kernel_id id);
};

}}
