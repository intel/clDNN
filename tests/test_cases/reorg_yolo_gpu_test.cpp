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
#include "api/CPP/reorg_yolo.hpp"
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

TEST(reorg_yolo_gpu_f32, reorg_yolo_test)
{
    //  reorg yolo test
    //  Input: bfyx:1x64x26x26
    //  Input: yolo_reorg_test_data.cpp

    extern std::vector<float> yolo_reorg_input;
    extern std::vector<float> yolo_reorg_ref;
    engine engine;
    const auto inpute_size = 64 * 26 * 26;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 64, 26, 26 } });

    set_values(input, yolo_reorg_input);

    topology topology(
        input_layout("input", input.get_layout()),
        reorg_yolo("reorg_yolo", "input", 2));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorg_yolo");

    auto output = outputs.begin()->second.get_memory();

    float epsilon = 0.00001f;
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < inpute_size; i++)
    {
        EXPECT_NEAR(yolo_reorg_ref[i], output_ptr[i], epsilon);
    }
    
}
