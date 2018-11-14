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

#pragma once

#include <string>
#include <mutex>
#include "auto_tuner.h"
#include "kernel_selector_common.h"

namespace kernel_selector 
{
    //72 compute units GT4e
    void tuning_cache_72(tuning_data&);
    void tuning_cache_72_1(tuning_data&);
    void tuning_cache_72_2(tuning_data&);
    void tuning_cache_72_3(tuning_data&);
    //24 Compute units GT2
    void tuning_cache_24(tuning_data&);
    void tuning_cache_24_1(tuning_data&);
    void tuning_cache_24_2(tuning_data&);
    void tuning_cache_24_3(tuning_data&);
    //48 Compute units GT3e 
    void tuning_cache_48(tuning_data&);
    void tuning_cache_48_1(tuning_data&);
    void tuning_cache_48_2(tuning_data&);
    void tuning_cache_48_3(tuning_data&);
    //64 Compute units ICL GT2
    void tuning_cache_64(tuning_data&);
    void tuning_cache_64_1(tuning_data&);
    //APL 
    void tuning_cache_18(tuning_data&);
    //APL E3930.
    void tuning_cache_12(tuning_data&);

    class auto_tuner_offline
    {
    private:
        static std::shared_ptr<auto_tuner_offline> instance;
        static std::mutex mutex;
        auto_tuner_offline() = delete;
        // this is singleton implementation, if called twice with different parameter, 
        // second call param will be ignored
        auto_tuner_offline(const uint32_t computeUnitsCount);
        tuning_data t_data;

        const std::map<uint32_t, void(*)(tuning_data&)> sku_cache_fillers
        {
            { 72 , tuning_cache_72 },
            { 24 , tuning_cache_24 },
            { 48 , tuning_cache_48 },
            { 64 , tuning_cache_64 },
            { 18 , tuning_cache_18 },
            { 12 , tuning_cache_18 },
        };

    public:
        static std::shared_ptr<auto_tuner_offline> get_instance(const uint32_t computeUnitsCount);
        tuning_data get_tuning_data() const { return t_data; }
   };
}