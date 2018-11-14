/*
// Copyright (c) 2018 Intel Corporation
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

#include "auto_tuner.h"
#include "auto_tuner_offline.h"
namespace kernel_selector 
{
    std::shared_ptr<auto_tuner_offline> auto_tuner_offline::instance = 0;
    std::mutex auto_tuner_offline::mutex;

    auto_tuner_offline::auto_tuner_offline(const uint32_t compute_units_count)
    {
        uint32_t temp_compute_units_count = compute_units_count;
        // TODO: this is temporary solution of cases where user has non-tuned configuration. needs to implement better logic
        // i.e. create table with number of eu's configuration that will point to common cache.
        if (compute_units_count == 0)
            temp_compute_units_count = 24;
        sku_cache_fillers.at(temp_compute_units_count)(t_data);
    }

    std::shared_ptr<auto_tuner_offline> auto_tuner_offline::get_instance(const uint32_t computeUnitsCount)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (instance == nullptr)
        {
            instance = std::make_shared<auto_tuner_offline>(auto_tuner_offline(computeUnitsCount));
        }
        return instance;
    }
}