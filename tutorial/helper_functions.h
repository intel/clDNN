/*
// Copyright (c) 2017 Intel Corporation
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
#include <initializer_list>

#include <../api/CPP/cldnn_defs.h>
#include <../api/CPP/memory.hpp>



using namespace cldnn;
template<typename T>
void set_values(const cldnn::memory& mem, std::initializer_list<T> args) {
    auto ptr = mem.pointer<T>();
    auto it = ptr.begin();
    for (auto x : args)
        *it++ = x;
}
