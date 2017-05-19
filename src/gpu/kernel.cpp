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

void kernel_execution_options::set_local_sizes()
{
    const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1 };
    auto total_lws = std::accumulate(std::begin(_local), std::end(_local), size_t(1), std::multiplies<size_t>());
    assert(total_lws != 0 && total_lws <= 256);

    for (auto i = _local.size(); i < _global.size(); ++i)
    {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while(rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

        while (_global[i] % optimal_lws_values[lws_idx]) lws_idx++;

        _local.push_back(optimal_lws_values[lws_idx]);
        total_lws *= optimal_lws_values[lws_idx];
    }
}
} }