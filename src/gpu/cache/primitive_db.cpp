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
#include "primitive_db.h"
#include <assert.h>
#include <algorithm>

namespace neural { namespace gpu { namespace manager {

primitive_db::primitive_db() : primitives({
    #include "primitive_db.inc"
}) { }

std::vector<code> primitive_db::get(const primitive_id & id)
{
    auto codes = primitives.equal_range(id);
    std::vector<code> temp;
    std::for_each(codes.first, codes.second, [&](auto c){ temp.push_back(c.second); });
	assert(temp.size() > 0 && "There should be at least one implementation of primitive");
    return temp;
}

} } }
