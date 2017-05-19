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
#pragma once

#include "primitive_db.h"
#include "cache.h"

namespace neural { namespace gpu { namespace manager {

/// \brief Class that selects a best binary using ordering provided by cost model
///
struct primitive_selector
{
    primitive_selector( );

    cache::binary_data get(context* context, const jit& jit, const primitive_id& id);

private:
    cache::cache binary_cache;
    primitive_db db;
};

} } }