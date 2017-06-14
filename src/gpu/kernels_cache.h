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
#include "cache/primitive_db.h"

namespace cl {
class Kernel;
}

namespace neural {namespace gpu {
class gpu_toolkit;

class kernels_cache {
public:
    typedef std::string kernel_id;
    typedef std::vector<std::pair<std::string, std::string>> jit_definitions;
    typedef cl::Kernel kernel_type;

private:
    gpu_toolkit& _context;
    std::mutex _mutex;
    std::map<std::string, std::string> _kernel_codes;
    std::map<std::string, kernel_type> _kernels;
    manager::primitive_db _database;
    bool _modified = true;

    std::vector<std::string> get_program_source() const;
    friend class gpu_toolkit;
    explicit kernels_cache(gpu_toolkit& context);
    void build_program();

public:
    kernel_id create_kernel_from_template(const std::string& template_name, jit_definitions definitions = jit_definitions(), std::string kernel_name = std::string());
    kernel_type get_kernel(kernel_id id);
};

}}
