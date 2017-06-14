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
#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/pooling.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "api/CPP/reorder.hpp"

using namespace cldnn;
using namespace tests;

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz3x3_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 3x3
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,  format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,3,3 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.0f, output_ptr[0]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 0.0, -1.0,  0.5]
    //
    //  Expected output:
    //  [ 2.0,  1.5]
    //  [ 2.0,  1.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.0f, output_ptr[0]);
    EXPECT_EQ(1.5f, output_ptr[1]);
    EXPECT_EQ(2.0f, output_ptr[2]);
    EXPECT_EQ(1.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr2x2_i4x4x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  [-0.25,  1.00,  0.50,  0.25]
    //  [ 2.00,  1.50, -0.50, -0.75]
    //  [ 0.00, -1.00,  0.50,  0.25]
    //  [ 0.50, -2.00, -1.50, -2.50]
    //
    //  Expected output:
    //  [ 2.0,  0.5]
    //  [ 0.5,  0.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }));

    network network(engine, topology);
    set_values(input_prim, { -0.25f, 1.00f, 0.50f, 0.25f, 2.00f, 1.50f, -0.50f, -0.75f, 0.00f, -1.00f, 0.50f, 0.25f, 0.50f, -2.00f, -1.50f, -2.50f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.0f, output_ptr[0]);
    EXPECT_EQ(0.5f, output_ptr[1]);
    EXPECT_EQ(0.5f, output_ptr[2]);
    EXPECT_EQ(0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, basic_max_yxfb_f32_wsiz2x2_wstr1x1_i3x3x2x2_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: max
    //  Padding: none
    //
    //  Input data:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [-0.5,  0.5,  0.0]   [-1.5, -0.5,  0.0]
    //  [ 1.0, -1.0, -2.0]   [ 0.0, -1.0,  1.5]
    //  [-1.0, -0.5, -0.5]   [-2.0,  1.0, -0.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  0.0, -0.5]   [ 0.0,  0.5, -0.5]
    //  [-2.0, -1.0,  1.0]   [ 1.0, -1.0,  0.0]
    //  [-0.5, -1.0,  1.5]   [ 0.5, -0.5,  0.0]
    //
    //  Expected output:
    //  FM: 0 BATCH: 0       FM: 1 BATCH: 0
    //  [ 1.0,  0.5]         [ 0.0,  1.5]
    //  [ 1.0, -0.5]         [ 1.0,  1.5]
    //
    //  FM: 0 BATCH: 1       FM: 1 BATCH: 1
    //  [ 0.5,  1.0]         [ 1.0,  0.5]
    //  [-0.5,  1.5]         [ 1.0,  0.0]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 0.5f, -1.5f, 0.0f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f, -0.5f, 0.0f, -0.5f, 1.0f, -2.0f, 0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -2.0f, 1.0f, 1.5f, 0.0f, -1.0f, -0.5f, -2.0f, 0.5f, -0.5f, -1.0f, 1.0f, -0.5f, -0.5f, 1.5f, -0.5f, 0.0f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(1.0f, output_ptr[0]); EXPECT_EQ(0.0f, output_ptr[2]);
    EXPECT_EQ(0.5f, output_ptr[4]); EXPECT_EQ(1.5f, output_ptr[6]);
    EXPECT_EQ(1.0f, output_ptr[8]); EXPECT_EQ(1.0f, output_ptr[10]);
    EXPECT_EQ(-0.5f, output_ptr[12]); EXPECT_EQ(1.5f, output_ptr[14]);

    EXPECT_EQ(0.5f,  output_ptr[1]);  EXPECT_EQ(1.0f, output_ptr[3]);
    EXPECT_EQ(1.0f,  output_ptr[5]);  EXPECT_EQ(0.5f, output_ptr[7]);
    EXPECT_EQ(-0.5f, output_ptr[9]);  EXPECT_EQ(1.0f, output_ptr[11]);
    EXPECT_EQ(1.5f,  output_ptr[13]); EXPECT_EQ(0.0f, output_ptr[15]);
}

TEST(pooling_forward_gpu, offsets_max_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 1.5, -0.5]
    //  [   -1, 0.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0, 0, -1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.50f, -0.50f, -1.00f, 0.50f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ( 1.5f, output_ptr[0]);
    EXPECT_EQ(-0.5f, output_ptr[1]);
    EXPECT_EQ(-1.0f, output_ptr[2]);
    EXPECT_EQ( 0.5f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_max_yxfb_f32_wsiz2x2_wstr2x2_i3x3x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 1.5,  -0.5]
    //  [   1,  -0.5]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);

    set_values(input_prim, { 
        1.50f, -1.00f, -0.50f,
        1.00f, -1.00f, -1.00f,
       -1.00f, -1.00f, -0.50f
    });

    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(1.5f, get_value<float>(output_ptr, 0));
    EXPECT_EQ(-0.5f, get_value<float>(output_ptr, 1));
    //TODO !!!implement correct output size calculation!!!
    EXPECT_EQ(1.0f, get_value<float>(output_ptr, 3));
    EXPECT_EQ(-0.5f, get_value<float>(output_ptr, 4));
}

TEST(pooling_forward_gpu, basic_avg_yxfb_f32_wsiz2x2_wstr1x1_i3x3x1x1_nopad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 1x1
    //  Pool mode: avg
    //  Padding: none
    //
    //  Input data:
    //  [-0.5,  1.0,  0.5]
    //  [ 2.0,  1.5, -0.5]
    //  [ 4.0, -1.0,  3.5]
    //
    //  Expected output:
    //  [ 1.0,   0.625]
    //  [ 1.625, 0.875]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average,{ 1,1,2,2 },{ 1,1,1,1 }));

    network network(engine, topology);
    set_values(input_prim, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 4.0f, -1.0f, 3.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    
    EXPECT_EQ(1.0f,   output_ptr[0]);
    EXPECT_EQ(0.625f, output_ptr[1]);
    EXPECT_EQ(1.625f, output_ptr[2]);
    EXPECT_EQ(0.875f, output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_f32_wsiz2x2_wstr2x2_i2x2x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [ 0.375, -0.125]
    //  [ -0.25,  0.125]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(0.375f,  output_ptr[0]);
    EXPECT_EQ(-0.125f, output_ptr[1]);
    EXPECT_EQ(-0.25f,  output_ptr[2]);
    EXPECT_EQ(0.125f,  output_ptr[3]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_f32_wsiz2x2_wstr2x2_i3x3x1x1_zeropad) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: zero
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5,  2.5]
    //  [ padd, -1.0,  0.5,  3.0]
    //  [ padd,  0.5,  0.0, -8.0]
    //
    //  Expected output:
    //  [  0.375,    0.5]
    //  [ -0.125, -1.125]

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 3, 3 } });

    topology topology;
    topology.add(input_layout("input_prim", input_prim.get_layout()));
    topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }));

    network network(engine, topology);
    set_values(input_prim, { 1.5f, -0.5f, 2.5f, -1.0f, 0.5f, 3.0f, 0.5f, 0.0f, -8.0f });
    network.set_input_data("input_prim", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "pool_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    EXPECT_EQ(0.375f,  output_ptr[0]);
    EXPECT_EQ(0.5f,    output_ptr[1]);
    //TODO !!!implement correct output size calculation!!!
    EXPECT_EQ(-0.125f, output_ptr[3]);
    EXPECT_EQ(-1.125f, output_ptr[4]);
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 0.375, -0.125, 0, 0]
    //  [ 0, 0, -0.25,  0.125, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 2, 2 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(pooling("pool_prim", "input_prim", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, { { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.375f, -0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.25f, 0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1.5, -0.5, 0, 0]
    //  [0, 1, -0.5, 0, 0]
    //  [0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 3, 3 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(pooling("pool_prim", "input_prim", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, { { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.50f, -1.00f, -0.50f,
            1.00f, -1.00f, -1.00f,
            -1.00f, -1.00f, -0.50f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f,-0.5f, 0.0f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Out Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd]
    //  [ padd,  1.5, -0.5, padd]
    //  [ padd, -1.0,  0.5, padd]
    //  [ padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 0.375, -0.125, 0, 0]
    //  [ 0, 0, -0.25,  0.125, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 2, 2 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ {0,0,1,2}, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, { { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, { 1.5f, -0.5f, -1.0f, 0.5f });
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.375f, -0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, -0.25f, 0.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, offsets_max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : -1x-1
    //  Input data:
    //  [ padd, padd, padd, padd, padd]
    //  [ padd,  1.5, -1.0, -0.5, padd]
    //  [ padd,  1.0, -1.0, -1.0, padd]
    //  [ padd, -1.0, -1.0, -0.5, padd]
    //  [ padd, padd, padd, padd, padd]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0]
    //  [0, 1.5, 0, 0, 0]
    //  [0, 1, -0.5, 0, 0]
    //  [0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 3, 3 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ { 0, 0, 1, 2 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, { { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.50f, -1.00f, -0.50f,
            1.00f, -1.00f, -1.00f,
            -1.00f, -1.00f, -0.50f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.5f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.f, -0.5f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, avg_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i2x2x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: avg
    //  Out Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 0x0
    //  Input data:
    //  [ 1, 2, 3, 4]
    //  [ 5,  1.5, -0.5, 6]
    //  [ 7, -1.0,  0.5, 8]
    //  [ 9, 10, 11, 12]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [ 0, 0, 2.375, 3.125, 0, 0]
    //  [ 0, 0, 6.25,  7.875, 0, 0]
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 4, 4 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::average, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,0,0 }, { { 0,0,2,2 }, 0 }));

        network network(engine, topology);
        set_values(input_prim, {
            1.f, 2.f, 3.f, 4.f,
            5.f, 1.5f, -0.5f, 6.f,
            7.f, -1.0f, 0.5f, 8.f,
            9.f, 10.f, 11.f, 12.f});
        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 2.375f, 3.125f, 0.0f, 0.0f,
            0.0f, 0.0f, 6.25f, 7.875f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}

TEST(pooling_forward_gpu, max_yxfb_bfyx_f32_wsiz2x2_wstr2x2_i3x3x1x1_inpad2x1_outpad2) {
    //  Brief test description.
    //
    //  Pool window: 2x2
    //  Pool stride: 2x2
    //  Pool mode: max
    //  Padding: 2x2
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //
    //  Input offset : 0x0
    //  Input data:
    //  [ 1, 2, 3, 4, 5]
    //  [ 6,  1.5, -1.0, -0.5, 7]
    //  [ 8,  1.0, -1.0, -1.0, 9]
    //  [ 10, -1.0, -1.0, -0.5, 11]
    //  [ 12, 13, 14, 15, 16]
    //
    //  Expected output:
    //  [0, 0, 0, 0, 0, 0]
    //  [0, 1, 3, 5, 0, 0]
    //  [0, 8, 1.5, 9, 0, 0]
    //  [0, 12, 14, 16, 0, 0]
    //  [0, 0, 0, 0, 0, 0]

    engine engine;
    std::vector<format> formats_to_test = { format::yxfb , format::bfyx };

    for (std::vector<format>::iterator it = formats_to_test.begin(); it != formats_to_test.end(); ++it)
    {
        std::cout << "Testing format: " << format::order(*it) << std::endl;

        tensor input_tensor( 1, 1, 5, 5 );
        auto input_prim = memory::allocate(engine, { data_types::f32, *it, input_tensor });

        topology topology;
        topology.add(input_layout("input_prim", input_prim.get_layout()));
        topology.add(reorder("reorder", "input_prim", input_prim.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })));
        topology.add(pooling("pool_prim", "reorder", pooling_mode::max, { 1,1,2,2 }, { 1,1,2,2 }, { 0,0,-1,-1 }, { { 0,0,1,1 }, 0 }));

        network network(engine, topology);

        set_values(input_prim, {
            1.f, 2.f, 3.f, 4.f, 5.f,
            6.f, 1.50f, -1.00f, -0.50f, 7.f,
            8.f, 1.00f, -1.00f, -1.00f, 9.f,
            10.f, -1.00f, -1.00f, -0.50f, 11.f,
            12.f, 13.f, 14.f, 15.f, 16.f
        });

        network.set_input_data("input_prim", input_prim);

        std::vector<float> expected = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.f, 3.f, 5.f, 0.0f, 0.0f,
            0.0f, 8.f, 1.5f, 9.f, 0.0f, 0.0f,
            0.0f, 12.f, 14.f, 16.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "pool_prim");

        auto output_prim = outputs.begin()->second.get_memory();
        auto output_ptr = output_prim.pointer<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], output_ptr[i]);
        }
    }
}