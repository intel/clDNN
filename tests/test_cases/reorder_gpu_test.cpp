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
#include "api/CPP/reorder.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>

#include <cmath>
#include <gmock/gmock.h>
#include <limits>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(reorder_gpu_f32, basic)
{
    //  Input               : yxfb:2x2x2x2
    //  Output              : bfyx:2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Output:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2,2,2,2 });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(reorder_gpu_f32, basic_subtract) {
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //  Subtract            : 1x2x2x2 (only first batch is taken into consideration)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Subtract:
    //  f0: b0:  1    1.5
    //  f0: b0:  2    2.5
    //  f1: b0:  4    3
    //  f1: b0:  2    1
    //
    //
    //  Output:
    //  b0 f0:  0    0.5
    //  b0 f0:  1    1.5
    //
    //  b0 f1:  1    3
    //  b0 f1:  5    7
    //
    //  b1 f0: -1   -1.5
    //  b1 f0: -1.5 -3
    //
    //  b1 f1: -2.5  2.2
    //  b1 f1: 10    7
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,  format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout( data_types::f32, format::bfyx, {2,2,2,2} );
    auto subtract = memory::allocate(engine, { data_types::f32, format::byxf, { 1, 2, 2, 2 } });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    topology topology(
        input_layout("input", input.get_layout()),
        input_layout("subtract", subtract.get_layout()),
        reorder("reorder", "input", output_layout, "subtract"));

    network network(engine, topology);
    network.set_input_data("input", input);
    network.set_input_data("subtract", subtract);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = { 0.0f,  0.5f,
                          1.0f,  1.5f,

                          1.0f,  3.0f,
                          5.0f,  7.0f,

                         -1.0f, -1.5f,
                         -1.5f, -3.0f,

                         -2.5f,  2.2f,
                         10.0f,  7.0f
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_f32, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2,2,2,2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()), reorder("reorder", "input", output_layout, subtract_val));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = { 0.5f, 1.5f,
                          2.5f, 3.5f,

                          2.5f, 3.5f,
                          4.5f, 5.5f,

                         -0.5f, -0.5f,
                          0.0f, -1.0f,

                         -1.0f,  2.7f,
                          9.5f,  5.5f
    };

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f16, basic_subtract_f32_output_f32) {
    //  Input               : 2x2x2x2 (FP16)
    //  Output              : 2x2x2x2 (FP32)
    //  Subtract            : 1x2x2x2 (FP32, only first batch is taken into consideration)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Subtract (FP32 - converted internally to FP16 before subtraction):
    //  f0: b0:  1    1.5
    //  f0: b0:  2    2.5
    //  f1: b0:  4    3
    //  f1: b0:  2    1
    //
    //
    //  Output:
    //  b0 f0:  0    0.5
    //  b0 f0:  1    1.5
    //
    //  b0 f1:  1    3
    //  b0 f1:  5    7
    //
    //  b1 f0: -1   -1.5
    //  b1 f0: -1.5 -3
    //
    //  b1 f1: -2.5  2.2
    //  b1 f1: 10    7
    //

    engine engine;

    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2,2,2,2 });
    auto subtract = memory::allocate(engine, { data_types::f32, format::byxf, { 1, 2, 2, 2 } });

    set_values(input, {
        half_t(0x3C00), half_t(0x0000), // 1.f, 0.f,
        half_t(0x4500), half_t(0x3E00), // 5.f, 1.5f,

        half_t(0x4000), half_t(0x0000), // 2.f, 0.f,
        half_t(0x4600), half_t(0x4533), // 6.f, 5.2f,

        half_t(0x4200), half_t(0x3800), // 3.f, 0.5f,
        half_t(0x4700), half_t(0x4A00), // 7.f, 12.f,

        half_t(0x4400), half_t(0xB800), // 4.f, -0.5f,
        half_t(0x4800), half_t(0x4800)  // 8.f, 8.f
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("subtract", subtract));
    topology.add(reorder("reorder", "input", output_layout, "subtract"));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = { 0.0f,  0.5f,
                          1.0f,  1.5f,

                          1.0f,  3.0f,
                          5.0f,  7.0f,

                         -1.0f, -1.5f,
                         -1.5f, -3.0f,

                         -2.5f,  2.2f,
                         10.0f,  7.0f
    };
    
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f16, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2 (FP16)
    //  Output              : 2x2x2x2 (FP16)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values (FP32 - converted internally to FP16 before subtraction)
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    engine engine;
    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f16, format::bfyx,{ 2,2,2,2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        half_t(0x3C00), half_t(0x0000), // 1.f, 0.f,
        half_t(0x4500), half_t(0x3E00), // 5.f, 1.5f,

        half_t(0x4000), half_t(0x0000), // 2.f, 0.f,
        half_t(0x4600), half_t(0x4533), // 6.f, 5.2f,

        half_t(0x4200), half_t(0x3800), // 3.f, 0.5f,
        half_t(0x4700), half_t(0x4A00), // 7.f, 12.f,

        half_t(0x4400), half_t(0xB800), // 4.f, -0.5f,
        half_t(0x4800), half_t(0x4800)  // 8.f, 8.f
    });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reorder("reorder", "input", output_layout, subtract_val));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    half_t answers[16] = { half_t(0x3800), half_t(0x3E00), //  0.5f, 1.5f,
                           half_t(0x4100), half_t(0x4300), //  2.5f, 3.5f,
                            
                           half_t(0x4100), half_t(0x4300), //  2.5f, 3.5f,
                           half_t(0x4480), half_t(0x4580), //  4.5f, 5.5f,
                            
                           half_t(0xB800), half_t(0xB800), // -0.5f, -0.5f,
                           half_t(0x0000), half_t(0xBC00), //  0.0f, -1.0f,
                            
                           half_t(0xBC00), half_t(0x4166), // -1.0f,  2.7f,
                           half_t(0x48C0), half_t(0x4580)  //  9.5f,  5.5f
    };

    auto output_ptr = output.pointer<half_t>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(static_cast<uint16_t>(answers[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}

TEST(reorder_gpu, basic_convert_f16_f32_f16) {
    //  Converts entire unambiguous range of FP16 numbers to FP32 and back.
    //
    //  Input               : 2x2x15873x1 (FP16)
    //  Intermediate        : 1x2x2x15873 (FP32) {different mem format but the same ordering because batch is 1}
    //  Output              : 2x2x15673x1 (FP16)
    //
    //  Output is expected to contain the same value as input in range of indices from 0x0000 to 0xF801.
    //

    engine engine;

    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    std::vector<half_t> expected_values;
    expected_values.resize(0xF804);
    for (int i = 0; i < 0x7C00; ++i)
        expected_values[i] = half_t(i);          // norms/denorms/zero (positive).
    for (int i = 0x7C00; i < 0xF800; ++i)
        expected_values[i] = half_t(i + 0x0400); // norms/denorms (negative).
    expected_values[0x7C00] = half_t(0x0000);    // NOTE: do not do final test for negative 0 (-0).
    // Special values.
    expected_values[0xF800] = half_t(0x7C00);    // +infinity
    expected_values[0xF801] = half_t(0xFC00);    // -infinity
    // Special values (ambiguous ones).
    expected_values[0xF802] = half_t(0x8000);    // -0
    expected_values[0xF803] = half_t(0xFC12);    // A NaN (sample: -NaN.0x12).

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, static_cast<int32_t>(expected_values.size()) / 4, 2, 2 } });
    layout interm_layout( data_types::f32, format::byxf, { 1, static_cast<int32_t>(expected_values.size()) / 4, 2, 2 });
    auto output_layout = input.get_layout();

    set_values(input, expected_values);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reorder("reorder_f16_f32", "input", interm_layout));
    topology.add(reorder("reorder_f32_f16", "reorder_f16_f32", output_layout));

    network network(
                                engine,
                                topology,
                                {
                                    build_option::outputs({"reorder_f16_f32", "reorder_f32_f16"})
                                });

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(2));
    EXPECT_TRUE(outputs.find("reorder_f16_f32") != outputs.end());
    EXPECT_TRUE(outputs.find("reorder_f32_f16") != outputs.end());

    auto interm = outputs.at("reorder_f16_f32").get_memory();
    auto interm_ptr = interm.pointer<float>();

    // Sample positive.
    EXPECT_TRUE(are_equal(interm_ptr[0x3400], 0.25f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3800], 0.5f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3C00], 1.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4000], 2.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4400], 4.0f));
    // Sample negative.
    EXPECT_TRUE(are_equal(interm_ptr[0x3400 + 0x7C00], -0.25f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3800 + 0x7C00], -0.5f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3C00 + 0x7C00], -1.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4000 + 0x7C00], -2.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4400 + 0x7C00], -4.0f));
    // Special values.
    EXPECT_TRUE(are_equal(interm_ptr[0xF800], std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(are_equal(interm_ptr[0xF801], -std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(are_equal(interm_ptr[0xF802], -0.0f));
    EXPECT_TRUE(std::isnan(interm_ptr[0xF803]));

    auto output = outputs.at("reorder_f32_f16").get_memory();
    auto output_ptr = output.pointer<half_t>();
    for (int i = 0; i < 0xF802; ++i) // NOTE: do not test for possibly ambiguous values of floating point (-0, NaNs).
    {
        EXPECT_TRUE(are_equal(static_cast<uint16_t>(expected_values[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}

TEST(reorder_gpu_f32, basic_yxfb_to_bfyx_input_padding)
{
    //  Input               : yxfb:2x2x2x2
    //  Output              : bfyx:2x2x2x2
    //
    //  Input:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //
    //  Output:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx, { 2,2,2,2 });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().format, input.get_layout().data_type, "", { { 0, 0, 1, 2 }, 0 }),
        reorder("reorder2", "reorder", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder2");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    };
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(reorder_gpu_f32, basic_bfyx_to_yxfb_input_padding)
{
    //  Input               : bfyx:2x2x2x2
    //  Output              : yxfb:2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Output:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::yxfb, { 2,2,2,2 });

    set_values(input, {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().format, input.get_layout().data_type, "", { { 0, 0, 2, 1 }, 0 }),
        reorder("reorder2", "reorder", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder2");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    };
    std::vector<float> out;
    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 16; i++)
    {
        out.push_back(output_ptr[i]);
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(reorder_gpu_opt, basic_remove_redundant)
{
    engine eng;

    memory in = memory::allocate(eng, { data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in.get_layout()),
        reorder("r1", "in", format::bfyx, data_types::f32),
        reorder("r2", "r1", format::yxfb, data_types::f32)
    };

    build_options opts;
    opts.set_option(build_option::debug(true)); //to enable query for optimized memory
    opts.set_option(build_option::optimize_data(true));

    network net(eng, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();

    EXPECT_TRUE(outputs.count("r1") == 0);
    ASSERT_TRUE(outputs.count("r2") == 1);
    EXPECT_TRUE(outputs.at("r2").get_memory().get_layout().format == format::yxfb);
}

TEST(reorder_gpu_opt, basic_remove_redundant_due_to_implicit_reorders)
{
    engine eng;

    memory in = memory::allocate(eng, { data_types::f32, format::yxfb, tensor{ 1, 2, 2, 1 } });
    memory weights = memory::allocate(eng, { data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in.get_layout()),
        convolution("conv", "in", { "weights" }),
        data("weights", weights),
        reorder("r1", "conv", format::bfyx, data_types::f32) //optimize data should add conversion from yxfb to bfyx and 'conv' should output data in bfyx as well (IE case)
    };

    build_options opts;
    opts.set_option(build_option::debug(true)); //to enable query for optimized memory
    opts.set_option(build_option::optimize_data(true));

    network net(eng, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();

    EXPECT_TRUE(outputs.count("r1") == 0);
    ASSERT_TRUE(outputs.count("conv") == 1);
    EXPECT_TRUE(outputs.at("conv").get_memory().get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, basic_remove_redundant_output_due_to_implicit_reorders)
{
    engine eng;

    memory in = memory::allocate(eng, { data_types::f32, format::yxfb, tensor{ 1, 2, 2, 1 } });
    memory weights = memory::allocate(eng, { data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in.get_layout()),
        convolution("conv", "in",{ "weights" }),
        data("weights", weights),
        reorder("r1", "conv", format::bfyx, data_types::f32) //optimize data should add conversion from yxfb to bfyx and 'conv' should output data in bfyx as well (IE case)
    };

    build_options opts;

    //we need to check if r1 will be successfully opimized and still we should be able to query for r1's output which should point to conv's output (note conv cannot be marked as output in this case)
    opts.set_option(build_option::outputs({ "r1" }));
    opts.set_option(build_option::optimize_data(true));

    network net(eng, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();

    EXPECT_TRUE(outputs.count("conv") == 0);
    ASSERT_TRUE(outputs.count("r1") == 1);
    EXPECT_TRUE(outputs.at("r1").get_memory().get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, non_trivial_remove_redundant)
{
    engine eng;

    memory in = memory::allocate(eng, { data_types::f32, format::yxfb, tensor{ 1, 1, 5, 2 } });
    topology tpl{
        input_layout("in", in.get_layout()),
        reorder("r1", "in", format::bfyx, data_types::f32)
    };

    build_options opts;

    opts.set_option(build_option::debug(true));
    opts.set_option(build_option::optimize_data(true));

    network net(eng, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();

    ASSERT_TRUE(outputs.count("in") == 1);
    ASSERT_TRUE(outputs.count("r1") == 1);
    EXPECT_TRUE(outputs.at("r1").get_memory().is_the_same_buffer(outputs.at("in").get_memory()));
    EXPECT_TRUE(outputs.at("in").get_memory().get_layout().format == format::yxfb);
    EXPECT_TRUE(outputs.at("r1").get_memory().get_layout().format == format::bfyx);
}

using namespace cldnn;

class reorder_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        for (auto generic_params : all_generic_params)
        {
            delete generic_params;
        }
        for (auto test_param : all_test_params)
        {
            auto primitive = std::get<1>(test_param);
            delete primitive;
        }
    }


    static std::vector<std::tuple<test_params*, cldnn::primitive*>> generate_specific_test_params()
    {
        generic_test::generate_generic_test_params(all_generic_params);

        const std::vector<cldnn::data_types> data_types = { cldnn::data_types::f32 ,  cldnn::data_types::f16 };

        for (auto test_param : all_generic_params)
        {
            cldnn::tensor input_tensor = test_param->input_layouts[0].size;

            std::vector<cldnn::layout> output_layouts = {};

            for (auto dt : data_types)
            {
                for (auto fmt : generic_test::test_input_formats)
                {
                    output_layouts.push_back({ dt, fmt, input_tensor });
                }
            }
            // TODO: check unsupported formats.

            //TODO: check subtract.
            std::vector<float> subtract = {};

            for (auto output_layout : output_layouts)
            {
                //TODO: check input + output padding.
                all_test_params.push_back(std::make_tuple(test_param, new reorder("reorder", "input0", output_layout, subtract)));
            }
        }

        return all_test_params;
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        return (    (format == cldnn_format_type::cldnn_format_yxfb) ||
                    (format == cldnn_format_type::cldnn_format_byxf) ||
                    (format == cldnn_format_type::cldnn_format_bfyx) ||
                    (format == cldnn_format_type::cldnn_format_fyxb)
                );
    }

    template<typename InputType, typename OutputType>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        const cldnn::reorder* reorder = (cldnn::reorder*)layer_params;
        primitive_id mean = reorder->mean;
        std::vector<float> subtract_per_feature = reorder->subtract_per_feature;
        assert(mean == "");
        assert(subtract_per_feature.size() == 0);
        
        auto output = memory::allocate(engine, cldnn::layout(reorder->output_data_type, inputs[0].get_layout().format, inputs[0].get_layout().size));

        cldnn::pointer<InputType> input_mem = inputs[0].pointer<InputType>();
        cldnn::pointer<OutputType> output_mem = output.pointer<OutputType>();

        for (size_t i = 0; i < inputs[0].get_layout().get_linear_size(); i++)
        {
            // Write the output in the same order as the input with type conversion as needed.
            // The correct order will be checked in generic_test::compare_buffers.
            output_mem[i] = (OutputType)input_mem[i];
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            if (((cldnn::reorder*)layer_params)->output_data_type == data_types::f32)
            {
                return generate_reference_typed<float, float>(inputs);
            }
            else
            {
                return generate_reference_typed<float, FLOAT16>(inputs);
            }
        }
        else
        {
            if (((cldnn::reorder*)layer_params)->output_data_type == data_types::f32)
            {
                return generate_reference_typed<FLOAT16, float>(inputs);
            }
            else
            {
                return generate_reference_typed<FLOAT16, FLOAT16>(inputs);
            }
        }
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<std::tuple<test_params*, cldnn::primitive*>> all_test_params;

};

std::vector<tests::test_params*> reorder_test::all_generic_params = {};
std::vector<std::tuple<test_params*, cldnn::primitive*>> reorder_test::all_test_params = {};

TEST_P(reorder_test, DISABLED_test_all)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(REORDER,
                        reorder_test,
                        ::testing::ValuesIn(reorder_test::generate_specific_test_params()),
                        tests::generic_test::custom_param_name_functor());
