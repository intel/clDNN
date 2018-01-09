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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/region_yolo.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>

#include <cmath>
#include <gmock/gmock.h>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(region_yolo_gpu_f32, region_yolo_test)
{
    //  region yolo test
    //  Input: bfyx:1x125x13x13
    //  Input: yolo_region_test_data.cpp

    extern std::vector<float> yolo_region_input;
    extern std::vector<float> yolo_region_ref;
    engine engine;
    const auto inpute_size = 125 * 13 * 13;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 125, 13, 13 } });

    set_values(input, yolo_region_input);

    topology topology(
        input_layout("input", input.get_layout()),
        region_yolo("region_yolo", "input", 4, 20, 5));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "region_yolo");

    auto output = outputs.begin()->second.get_memory();

    float epsilon = 0.00001f;
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < inpute_size; i++)
    {
        EXPECT_NEAR(yolo_region_ref[i], output_ptr[i], epsilon);
    }

}
