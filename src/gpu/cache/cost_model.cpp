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
#include "cost_model.h"

namespace neural { namespace gpu { namespace cache {

cost_model::cost::cost(size_t value) : value{ value } { }

bool cost_model::cost::operator<(const cost & rhs) const
{
    return value < rhs.value;
}

cost_model::cost cost_model::rate(const binary_data & kernel_binary)
{
    return cost{ kernel_binary.length() };
}

} } }