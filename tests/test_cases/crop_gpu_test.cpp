/*
// Copyright (c) 2017 Intel Corporation
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
#include "api/CPP/crop.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t y, size_t x, int min, int max) {
    static std::default_random_engine generator(random_seed);
    int k = 8; // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int> distribution(k * min, k * max);
    std::vector<T> v(b*f*x*y);
    for (size_t i = 0; i < b*f*x*y; ++i) {
        v[i] = (T)distribution(generator);
        v[i] /= k;
    }
    return v;
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    engine engine;

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    engine engine;

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();
    std::vector<float> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15  
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13     
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    engine engine;

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}