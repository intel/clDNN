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
#include "api/CPP/convolution.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include <api/CPP/data.hpp>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>
#include <api/CPP/reorder.hpp>

using namespace cldnn;
using namespace tests;


namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}



template<typename T>
T kahan_summation(std::vector<T> &input) {
    T sum = 0;
    T c = 0;
    for (T x : input) {
        T y = x - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template<typename T>
VVF<T> reference_convolve(VVVF<T> &input, VVVF<T> &filter, int stride_y, int stride_x, float bias, int dilation_y = 1, int dilation_x = 1,
        int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0, 
        int output_padding_x = 0, size_t f_begin = 0)
{
    size_t kernel_extent_y = dilation_y * (filter[0].size() - 1) + 1;
    size_t kernel_extent_x = dilation_x * (filter[0][0].size() - 1) + 1;
    size_t output_y = 1 + (input[0].size() - kernel_extent_y + 2 * input_padding_y) / stride_y + 2 * output_padding_y;
    size_t output_x = 1 + (input[0][0].size() - kernel_extent_x + 2 * input_padding_x) / stride_x + 2 * output_padding_x;
    VVF<T> output(output_y, VF<T>(output_x, bias));
    for (size_t f = 0; f < filter.size(); ++f) {
        for (size_t y = 0; y < (output_y - 2 * output_padding_y); ++y) {
            for (size_t x = 0; x < (output_x - 2 * output_padding_x); ++x) {
                VF<T> values;
                values.reserve(filter[0].size() * filter[0][0].size());
                for (size_t yf = 0; yf < filter[0].size(); ++yf) {
                    int yi = -input_padding_y + (int)yf * dilation_y + stride_y * (int)y;
                    if (yi < 0 || (int)input[0].size() <= yi) continue;
                    for (size_t xf = 0; xf < filter[0][0].size(); ++xf) {
                        int xi = -input_padding_x + (int)xf * dilation_x + stride_x * (int)x;
                        if (xi < 0 || (int)input[0][0].size() <= xi) continue;
                        values.push_back(input[f_begin + f][yi][xi] * filter[f][yf][xf]);
                    }
                }
                output[y + output_padding_y][x + output_padding_x] += kahan_summation<T>(values);
            }
        }
    }
    return output;
}



// rounds floating point number, fraction precision should be in the range [0,23]
// masks the bits:
// 1 11111111 11111111111111100000000
// |      |            |
// sign  exp        fraction
float float_round(float x, size_t fraction_precision = 15) {
    uint32_t mask = ~((1 << (23 - fraction_precision)) - 1);
    reinterpret_cast<uint32_t&>(x) &= mask;
    return x;
}

template<typename T>
void print_2d(VVF<T> &data) {
    std::cout << "\nprinting data:\n";
    for (VF<T> &v : data) {
        for (float f : v) {
            std::cout << std::setw(4) << std::right << f << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static std::string print_test_params(cldnn::format test_input_fmt, cldnn::format test_filter_fmt, 
    int input_b, int input_f, int input_y, int input_x,
    int filter_o, int filter_i, int filter_y, int filter_x,
    int stride_y, int stride_x,
    int dilation_y, int dilation_x,
    int input_padding_y, int input_padding_x,
    int output_padding_y, int output_padding_x)
{
    std::stringstream str;
    
    str << std::endl
        << "failing test parameters:" << std::endl
        << "input_format = " << format::traits(test_input_fmt).order << std::endl
        << "filter_format = " << format::traits(test_filter_fmt).order << std::endl
        << "input_b = " << input_b << std::endl
        << "input_f = " << input_f << std::endl
        << "input_y = " << input_y << std::endl
        << "input_x = " << input_x << std::endl
        << "filter_o = " << filter_o << std::endl
        << "filter_i = " << filter_i << std::endl
        << "filter_y = " << filter_y << std::endl
        << "filter_x = " << filter_x << std::endl
        << "stride_y = " << stride_y << std::endl
        << "stride_x = " << stride_x << std::endl
        << "dilation_y = " << dilation_y << std::endl
        << "dilation_x = " << dilation_x << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl;

    return str.str();
}

template<typename T>
void generic_convolution_test(cldnn::format test_input_fmt, cldnn::format test_filter_fmt,
                              int input_b, int input_f, int input_y, int input_x,
                              int filter_o, int filter_i, int filter_y, int filter_x, 
                              int stride_y, int stride_x,
                              int dilation_y, int dilation_x,
                              int input_padding_y, int input_padding_x, 
                              int output_padding_y, int output_padding_x) {

    int min_random = -2, max_random = 2;
    int split = input_f / filter_i;

    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VVVVVF<T> filter_rnd = generate_random_5d<T>(split, filter_o, filter_i, filter_y, filter_x, min_random, max_random);
    VVF<T> bias_rnd_vec = generate_random_2d<T>(split, filter_o, min_random, max_random);
    VVVVF<T> output_cpu(input_b, VVVF<T>(filter_o * split));
    for (int output_b = 0; output_b < input_b; ++output_b) {
        for (int s = 0; s < split; ++s) {
            for (int output_f = 0; output_f < filter_o; ++output_f) {
                output_cpu[output_b][output_f + s * filter_o] = reference_convolve<T>(input_rnd[output_b], filter_rnd[s][output_f], stride_y, stride_x, 
                    bias_rnd_vec[s][output_f], dilation_y, dilation_x, input_padding_y, input_padding_x, output_padding_y, output_padding_x, s * filter_i);
            }
        }
    }

    engine engine;
    
    tensor input_tensor( input_b, input_f, input_x, input_y );
    tensor filter_tensor( filter_o, filter_i, filter_x, filter_y );

    auto input = memory::allocate(engine, { cldnn::type_to_data_type<T>::value, test_input_fmt, input_tensor } );
    std::vector<memory> weights, biases;
    weights.reserve(split);
    biases.reserve(split);
    for (int s = 0; s < split; ++s) {
        weights.push_back(memory::allocate(engine, { cldnn::type_to_data_type<T>::value, test_filter_fmt, filter_tensor }));
        biases.push_back(memory::allocate(engine, { cldnn::type_to_data_type<T>::value, format::bfyx,{ filter_o,1,1,1 } }));
    }

    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
    set_values(input, input_rnd_vec);
    for (int s = 0; s < split; ++s) {
        VF<T> filter_rnd_vec = flatten_4d<T>(test_filter_fmt, filter_rnd[s]);
        set_values(weights[s], filter_rnd_vec);
        set_values(biases[s], bias_rnd_vec[s]);
    }

    std::vector<std::string> weights_str(split), biases_str(split);
    for (int s = 0; s < split; ++s) {
        weights_str[s] = "weights" + std::to_string(s);
        biases_str[s] = "biases" + std::to_string(s);
    }

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout()), // This node is needed for optimize_data flag
        convolution(
            "conv",
            "reorder",
            weights_str,
            biases_str,
            { 1,1,stride_x, stride_y },
            { 0, 0, -input_padding_x, -input_padding_y },
            { 1,1,dilation_x, dilation_y },
            false,
            0,
            padding{ { 0,0,output_padding_x, output_padding_y }, 0 })
    );

    for (int s = 0; s < split; ++s) {
        topology.add(data(weights_str[s], weights[s]));
        topology.add(data(biases_str[s], biases[s]));
    }
    cldnn::build_options options;
    if (input.get_layout().format == cldnn::format::bfyx)
    {
        options.set_option(cldnn::build_option::optimize_data(true));
    }

    std::map<primitive_id, network_output> outputs;
    ASSERT_NO_THROW(
    {
        network network(engine, topology, options);
        network.set_input_data("input", input);
        outputs = network.execute();
    }) << print_test_params(test_input_fmt, test_filter_fmt, input_b, input_f, input_y, input_x, filter_o, filter_i, filter_y, filter_x, 
                            stride_y, stride_x, dilation_y, dilation_x, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    EXPECT_EQ(output_layout.format.value, test_input_fmt.value);
    tensor output_tensor = output_layout.get_buffer_size();
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(y_size, (int)output_cpu[0][0].size());
    EXPECT_EQ(x_size, (int)output_cpu[0][0][0].size());
    EXPECT_EQ(f_size, (int)output_cpu[0].size());
    EXPECT_EQ(b_size, (int)output_cpu.size());
    bool test_is_correct = true;
    const auto output_desc = generic_test::get_linear_memory_desc(output_layout);

    for (int b = 0; b < b_size; b++)
    {
        for (int f = 0; f < f_size; f++)
        {
            for (int y = 0; y < y_size; y++)
            {
                for (int x = 0; x < x_size; x++)
                {
                    if ( (y < output_padding_y) ||
                         (y >= y_size - output_padding_y) ||
                         (x < output_padding_x) ||
                         (x >= x_size - output_padding_x) )
                    {
                        //Ignore padded values in the output.
                        continue;
                    }
                    size_t res_index = generic_test::get_linear_index(output_layout, b, f, y, x, output_desc);
                    // CPU reference format is bfyx.
                    if (!floating_point_equal(output_cpu[b][f][y][x], output_ptr[res_index])) {
                        test_is_correct = false;
                        break;
                    }
                }
            }
        }
    }
    EXPECT_EQ(test_is_correct, true) << print_test_params(test_input_fmt, test_filter_fmt, input_b, input_f, input_y, input_x, filter_o, filter_i, filter_y, filter_x, 
                                                          stride_y, stride_x, dilation_y, dilation_x, input_padding_y, input_padding_x, output_padding_y, output_padding_x);

    bool print_data = !test_is_correct;
    if (!print_data) return;

    std::cout << "\ninput:\n\n";
    for (int b = 0; b < input_b; ++b) {
        for (int f = 0; f < input_f; ++f) {
            std::cout << "b" << b << ", f" << f << ":\n";
            print_2d(input_rnd[b][f]);
        }
    }
    std::cout << "\nfilter:\n\n";
    for (int s = 0; s < split; ++s) {
        for (int o = 0; o < filter_o; ++o) {
            for (int i = 0; i < filter_i; ++i) {
                std::cout << "s" << s << " o" << o << ", i" << i << ":\n";
                print_2d(filter_rnd[s][o][i]);
            }
            std::cout << "s" << s << " o" << o << " bias = " << (float)bias_rnd_vec[s][o] << "\n\n";
        }
    }

    VVVVF<T> output_gpu(b_size, VVVF<T>(f_size, VVF<T>(y_size, VF<T>(x_size))));
    for (int b = 0; b < b_size; ++b) {
        for (int f = 0; f < f_size; ++f) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    output_gpu[b][f][y][x] = output_ptr[b + b_size * (f + f_size * (x + x_size * y))];
                }
            }
        }
    }
    std::cout << "\ngpu test output:\n";
    for (int b = 0; b < b_size; ++b) {
        for (int f = 0; f < f_size; ++f) {
            std::cout << "b" << b << " f" << f << ":\n";
            print_2d(output_gpu[b][f]);
        }
    }
    
    std::cout << "\ncpu reference test output:\n";
    for (int b = 0; b < b_size; ++b) {
        for (int f = 0; f < f_size; ++f) {
            std::cout << "b" << b << " f" << f << ":\n";
            print_2d(output_cpu[b][f]);
        }
    }
}

struct input_filter_format_pair {
    format input;
    format filter;
    bool is_fp32;
    bool is_fp16;
};

static input_filter_format_pair test_formats[] = {
      // input      filter        is fp32  is fp16
    { format::bfyx, format::bfyx, true,    false },
    { format::yxfb, format::yxfb, true,    true },
    { format::bfyx, format::bfyx, true,    false },
    { format::bfyx, format::yxfb, true,    true },
};

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

// Test that should cover different combinations of parameters.
// It uses a cpu convolution implementation developed for tests only.
// Note: to debug a failing combination more easily try:
//    * Isolate the problematic combination of parameters.
//    * Decrease the random values range in generic_convolution_test().
//    * Set k = 1 inside the function generate_random_1d().
TEST(DISABLED_convolution_gpu, generic_random) {
    std::vector<std::pair<int, int>> input_sizes = { { 100, 100 },{ 227, 227 },{ 400, 600 },{ 531, 777 },{ 4096, 1980 } };
    std::vector<std::pair<int, int>> filter_sizes = { { 1, 1 },{ 2, 2 },{ 3, 3 },{ 4, 4 },{ 5, 5 },{ 7, 7 },{ 9, 9 },{ 11, 11 },{ 1, 3 },{ 5, 2 } };
    std::vector<std::pair<int, int>> dilation_sizes = { { 1, 1 }, { 2, 4 } };

    engine engine;
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    for (int input_b = 1; input_b <= 1; input_b *= 2) {
        for (int input_f = 1; input_f <= 6; ++input_f) {
            for (std::pair<int, int> &input_yx : input_sizes) {
                for (int filter_o = 1; filter_o <= 2; ++filter_o) {
                    for (int filter_i = 1; filter_i <= input_f; ++filter_i) {
                        if (input_f % filter_i != 0)
                            continue; // only legal split is allowed
                        for (std::pair<int, int> &filter_yx : filter_sizes) {
                            for (std::pair<int, int> &dilation_yx : dilation_sizes) {
                                std::pair<int, int> kernel_extent_yx = std::make_pair(dilation_yx.first * (filter_yx.first - 1) + 1, dilation_yx.second * (filter_yx.second - 1) + 1);
                                if (input_yx.first < kernel_extent_yx.first || input_yx.second < kernel_extent_yx.second)
                                    continue; // the filter can't be larger than the input
                                for (int stride_y = 1; stride_y <= 5; ++stride_y) {
                                    for (int stride_x = 1; stride_x <= 5; ++stride_x) {
                                        for (int input_padding_y = 0; input_padding_y <= 1; ++input_padding_y) {
                                            for (int input_padding_x = 0; input_padding_x <= 1; ++input_padding_x) {
                                                for (int output_padding_y = 0; output_padding_y <= 1; ++output_padding_y) {
                                                    for (int output_padding_x = 0; output_padding_x <= 1; ++output_padding_x) {
                                                        for (int i = 0; i < (int)ARRAY_SIZE(test_formats); i++) {
                                                            if (test_formats[i].is_fp32) {
                                                                generic_convolution_test<float>(test_formats[i].input, test_formats[i].filter,
                                                                    input_b, input_f, input_yx.first, input_yx.second,    // BFYX
                                                                    filter_o, filter_i, filter_yx.first, filter_yx.second,
                                                                    stride_y, stride_x,
                                                                    dilation_yx.first, dilation_yx.second,
                                                                    input_padding_y, input_padding_x,
                                                                    output_padding_y, output_padding_x);
                                                            }
                                                            if (!f16_supported) continue;
                                                            if (test_formats[i].is_fp16) {
                                                                generic_convolution_test<FLOAT16>(test_formats[i].input, test_formats[i].filter,
                                                                    input_b, input_f, input_yx.first, input_yx.second,    // BFYX
                                                                    filter_o, filter_i, filter_yx.first, filter_yx.second,
                                                                    stride_y, stride_x,
                                                                    dilation_yx.first, dilation_yx.second,
                                                                    input_padding_y, input_padding_x,
                                                                    output_padding_y, output_padding_x);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_no_bias) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }, { 1,1,1,2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_convolution) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20
    //
    //  Bias:
    //  1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 21.0f, 28.0f, 39.0f },
        { 18.0f, 20.0f, 20.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution( "conv", "input", { "weights" }, { "biases" }, { 0,0,1,2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_convolution_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : 2x1
    //  Output : 6x5
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1
    //  2  4  6  8  5
    //  4  8 11 15  9
    //  6 11 12 16 10
    //  4  7  7  9  6
    //  1  1  1  1  1
    //
    //  Bias:
    //  1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,-1,-2 },
            { 1, 1, 1, 1 },
            false,
            0,
            padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 6);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_convolution_input_and_output_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : 2x1
    //  Output : 8x9
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1  1
    //  1  1  2  4  6  8  5  1  1
    //  1  1  4  8 11 15  9  1  1
    //  1  1  6 11 12 16 10  1  1
    //  1  1  4  7  7  9  6  1  1
    //  1  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    const int x_pad = 2;
    const int y_pad = 1;
    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,-1,-2 },
            { 1, 1, 1, 1 },
            false,
            0,
            padding{ { 0,0,-x_pad,-y_pad }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_size = output_layout.get_buffer_size();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_size.spatial[1];
    int x_size = output_size.spatial[0];
    int f_size = output_size.feature[0];
    int b_size = output_size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 8);
    EXPECT_EQ(x_size, 9);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = y_pad; y < y_size - y_pad; ++y) 
    {
        for (int x = x_pad; x < x_size - x_pad; ++x) 
        {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad_random) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //
    //  Filter
    //  rnd  rnd
    //  rnd  rnd
    //
    //  Bias
    //  rnd
    //
    //  Output:
    //  rnd  rnd
    //  rnd  rnd

    size_t batch = 1, input_f = 1, input_y = 4, input_x = 4;

    VVVVF<float> input_rnd = generate_random_4d<float>(batch, input_f, input_y, input_x, -10, 10);
    VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
    VVVVF<float> filter_rnd = generate_random_4d<float>(1, 1, 2, 2, -10, 10);
    VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);
    VF<float> bias_rnd = generate_random_1d<float>(1, -10, 10);
    VVVVF<float> output_rnd(batch, VVVF<float>(filter_rnd.size()));
    for (size_t b = 0; b < output_rnd.size(); ++b) {
        for (size_t of = 0; of < filter_rnd.size(); ++of) {
            output_rnd[b][of] = reference_convolve<float>(input_rnd[b], filter_rnd[of], 2, 2, bias_rnd[of]);
        }
    }
    VF<float> output_rnd_vec = flatten_4d<float>(format::yxfb, output_rnd);

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32,  format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32,  format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32,  format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", {"weights"}, {"biases"}, {1,1,2,2})
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    for (size_t i = 0; i < output_rnd.size(); ++i) {
        float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad_random) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  rnd  rnd    rnd  rnd
    //  rnd  rnd    rnd  rnd
    //
    //  Filter:
    //  rnd  rnd
    //  rnd  rnd
    //
    //  Bias:
    //  rnd
    //
    //  Output:
    //  rnd  rnd

    size_t batch = 2, input_f = 1, input_y = 2, input_x = 2;

    VVVVF<float> input_rnd = generate_random_4d<float>(batch, input_f, input_y, input_x, -10, 10);
    VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
    VVVVF<float> filter_rnd = generate_random_4d<float>(1, 1, 2, 2, -10, 10);
    VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);
    VF<float> bias_rnd = generate_random_1d<float>(1, -10, 10);
    VVVVF<float> output_rnd(batch, VVVF<float>(filter_rnd.size()));
    for (size_t b = 0; b < output_rnd.size(); ++b) {
        for (size_t of = 0; of < filter_rnd.size(); ++of) {
            output_rnd[b][of] = reference_convolve<float>(input_rnd[b], filter_rnd[of], 2, 2, bias_rnd[of]);
        }
    }
    VF<float> output_rnd_vec = flatten_4d<float>(format::yxfb, output_rnd);

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    for (size_t i = 0; i < output_rnd.size(); ++i) {
        float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  8  0.5
    //  6  9

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.5f, output_ptr[1]);
    EXPECT_FLOAT_EQ(6.0f, output_ptr[2]);
    EXPECT_FLOAT_EQ(9.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  0.5   1.5    2.3 -0.4
    //  2.0  -4.0    1.0  3.0
    //
    //  Filter:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias:
    //  -1
    //
    //  Output:
    //  3.65 -5.36
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 } )
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(3.65f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.36f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz2x1x2x1_in1x2x1_nopad) {
    //  Filter : 1x2x1x2x1
    //  Input  : 1x1x2x1
    //  Output : 1x2x1x1
    //
    //  Input:
    //  1.0    2.0
    //
    // Filter:
    //   1.0    2.0  ofm=0
    //  -1.0   -2.0  ofm=1
    //
    //  Bias:
    //  0.1 -0.2
    //
    //  Output:
    //   5.1  f=0
    //  -5.2  f=1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 1, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input, { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases, { 0.1f, -0.2f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,5,5 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(5.1f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.2f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz3x2x2x1_in2x2x1_nopad) {
    //  Filter : 1x3x2x2x1
    //  Input  : 1x2x2x1
    //  Output : 1x3x1x1
    //
    //  Input:
    //  1.0    2.0  f=0
    //  3.0    4.0  f=1
    //
    // Filter:
    //   1.0    2.0  ifm=0  ofm=0
    //   3.0    4.0  ifm=1
    //
    //   5.0    6.0  ifm=0  ofm=1
    //   7.0    8.0  ifm=1
    //
    //   9.0   10.0  ifm=0  ofm=2
    //  11.0   12.0  ifm=1
    //  Bias:
    //   -5     -6     -7
    //
    //  Output:
    //   25.0  f=0
    //   64,0  f=1
    //  103.0  f=2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 1, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 1 } });

    set_values(input, { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases, { -5.0f, -6.0f, -7.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,5,5 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(25.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(64.0f, output_ptr[1]);
    EXPECT_FLOAT_EQ(103.0f, output_ptr[2]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2x1x3
    //  Stride : 2x2
    //  Input  : 2x2x1x1
    //  Output : 1x1x3x1
    //
    //  Input:
    //  -2.3 -0.1
    //   3.1  1.9
    //
    //  Filter:
    //  -1.1  1.5       0.1  0.2        2.0  -1.0
    //   0.5 -0.5       0.4  0.7        2.5  -1.5
    //
    //  Bias:
    //  0.1 -0.2 0.3
    //
    //  Output:
    //   0.7
    //   2.12
    //   3.08

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 1 } });

    set_values(input, { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 1.5f, 0.5f, -0.5f, 0.1f, 0.2f, 0.4f, 0.7f, 2.0f, -1.0f, 2.5f, -1.5f });
    set_values(biases, { 0.1f, -0.2f, 0.3f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_TRUE(are_equal(3.08f, output_ptr[0]));
    EXPECT_TRUE(are_equal(2.12f, output_ptr[1]));
    EXPECT_TRUE(are_equal(0.7f,  output_ptr[2]));
}

TEST(convolution_f32_fw_gpu, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //  Filter  : 3x3
    //  Stride  : 2x2
    //  Input   : 2x2
    //  Output  : 1x1
    //  Padding : zero
    //
    //  Input:
    //  -0.5   1.0   padd
    //   0.5   2.0   padd
    //  padd  padd   padd
    //
    //  Filter
    //  -2    0.5  3.5
    //   1.5  4   -5
    //   0.5  1.5 -1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  12.25
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(12.25f, output_ptr[0]);
}

TEST(convolution_f32_fw_gpu, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //   Filter       : 3x3
    //   Stride       : 2x2
    //   Input        : 2x2
    //   Input offset : -1x-1
    //   Output       : 2x2
    //   Output offset: 1x1
    //   Padding      : zero
    //
    //   Input:
    //   padd padd  padd
    //   padd -0.5   1
    //   padd  0.5   2.0
    //
    //   Filter
    //   -2    0.5  3.5
    //    1.5  4   -5
    //    0.5  1.5 -1.5
    //
    //   Bias
    //   2
    //
    //   Output:
    //   rnd   rnd
    //   rnd   2.0
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,2,2 },
            { 0,0,-1,-1 },
            { 1, 1, 1, 1 },
            false,
            0,
            padding{ { 0,0,1,1 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-7.25f, output_ptr[4]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4x2
    //  Output : 2x2x2
    //
    //  Input:
    //  f0: -0.5   1     0.5  2
    //       1.5  -0.5   0   -1
    //       0.5   0.5  -1    1
    //       0.5   2     1.5 -0.5
    //
    //  f1:  0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //       0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  3.65 0.5 -5.36
    //   6  3.65 9   -5.36

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 0,0,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2) {
    //  2x Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x4x4x2
    //  Output : 2x2x2x2
    //
    //  Input:
    //  f0b0: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f0b1: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f1b0:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //  f1b1:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  8 3.65 3.65 0.5  0.5 -5.36 -5.36
    //   6  6 3.65 3.65 9    9   -5.36 -5.36

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
       -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 8));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 9));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 10));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 11));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 12));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 13));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 14));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 15));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2 but with batch 1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f,
        1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f,
        0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f,
        2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f,
        -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f,
        0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f,
        -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f,
        1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f,
        2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f,
        1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f,
        -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    for (uint32_t i = 0; i < 8; i++)
    {
        auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

        set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
        set_values(biases1, { 2.0f });
        set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
        set_values(biases2, { -1.0f });

        primitive_id weights_id = "weights_" + std::to_string(i);
        primitive_id weights2_id = "weights2_" + std::to_string(i);
        primitive_id bias_id = "biases_" + std::to_string(i);
        primitive_id bias2_id = "biases2_" + std::to_string(i);

        weights_vec.push_back(weights_id);
        weights_vec.push_back(weights2_id);
        bias_vec.push_back(bias_id);
        bias_vec.push_back(bias2_id);

        topology.add(
            data(weights_id, weights1),
            data(bias_id, biases1),
            data(weights2_id, weights2),
            data(bias2_id, biases2)
        );

    }

    topology.add(
        convolution(
            "conv",
            "input",
            weights_vec,
            bias_vec,
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f,
        0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f,
        6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f,
        9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt_bfyx) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2 but with batch 1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    for (uint32_t i = 0; i < 8; i++)
    {
        auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

        set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
        set_values(biases1, { 2.0f });
        set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
        set_values(biases2, { -1.0f });

        primitive_id weights_id = "weights_" + std::to_string(i);
        primitive_id weights2_id = "weights2_" + std::to_string(i);
        primitive_id bias_id = "biases_" + std::to_string(i);
        primitive_id bias2_id = "biases2_" + std::to_string(i);

        weights_vec.push_back(weights_id);
        weights_vec.push_back(weights2_id);
        bias_vec.push_back(bias_id);
        bias_vec.push_back(bias2_id);

        topology.add(
            data(weights_id, weights1),
            data(bias_id, biases1),
            data(weights2_id, weights2),
            data(bias2_id, biases2)
        );

    }

    topology.add(
        convolution(
            "conv",
            "input",
            weights_vec,
            bias_vec,
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  0.0
    //  f3: -0.5
    //
    //
    //  Filter1:
    //  -2 -0.5  ofm=0
    //   1  2    ofm=1 
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  1.5  ofm=0
    //   2  0.5  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2.25  
    //   7.5
    //
    //  -1.75
    //   2.25

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 4, 1, 1 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input, {
       1.5f, 0.5f, 0.0f, -0.5f
    });
    set_values(weights1, { -2.0f, -0.5f, 1.0f, 2.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 1.5f, 2.0f, 0.5f });
    set_values(biases2, { -1.0f, 2.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-2.25f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(7.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(-1.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(2.25f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x2x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x2
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //
    //  f1:  0.5
    //
    //  Filter1:
    //  -2  ofm=0
    //   1  ofm=1 
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  ofm=0
    //   2  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2  
    //   6.5
    //
    //   1
    //   3.5


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 1, 1 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input, {
        1.5f, 0.5f
    });
    set_values(weights1, { -2.0f, 1.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 2.0f });
    set_values(biases2, { -1.0f, 2.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(6.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(1.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.5f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_filter_1x3x2x1x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x6
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  2
    //  f3: -1.0
    //
    //  Filter1:
    //  -2   1   ofm=0
    //   1   3   ofm=1
    //   0.5 8   ofm=2
    //  Bias1:
    //   1   5   3
    //
    //  Filter2:
    //   4  -4   ofm=0
    //   2   0.5 ofm=1
    //  -0.5 3   ofm=2
    //
    //  Bias2:
    //  -1   2.5 2
    //
    //  Output:
    //  -1.5  
    //   8
    //   7.75
    //
    //   11
    //   6
    //  -2


    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 4, 1, 1 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 6 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights1, { -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f });
    set_values(biases1, { 1.0f, 5.0f, 3.0f });
    set_values(weights2, { 4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f });
    set_values(biases2, { -1.0f, 2.5f, 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-1.5f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(7.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(11.0f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 5));

}

TEST(convolution_gpu, trivial_convolution_relu) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  0.0
    //  2  5

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1, 1, 1, 1 },
            true,
            0)
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(0.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

TEST(convolution_gpu, relu_with_negative_slope) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //  Negative Slope : 0.1

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  -0.35
    //  2  5

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1, 1, 1, 1 },
            true,
            0.1f)
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(-0.35f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

TEST(convolution_gpu, basic_yxfb_4_4_yxfb_2_2_b16_if2_of16_st2_2_p0_sp1_fp32)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    const auto input_format   = format::yxfb;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = format::bfyx;
#else
    const auto weights_format = format::yxfb;
#endif
    const auto biases_format = format::bfyx;

    const int32_t batch_size = 16;
    const int32_t input_feature_count = 2;
    const int32_t output_feature_count = 16;

    const int32_t stride_x = 2;
    const int32_t stride_y = 2;

    const int32_t input_x = 4;
    const int32_t input_y = 4;
    const int32_t weights_x = 2;
    const int32_t weights_y = 2;
    const int32_t output_x = (input_x - weights_x) / stride_x + 1;
    const int32_t output_y = (input_y - weights_y) / stride_y + 1;

    engine engine;

    auto input_size = tensor( batch_size, input_feature_count, input_x, input_y );
    auto input = memory::allocate(engine, { data_types::f32, input_format, input_size });
    auto weights_size = tensor( output_feature_count, input_feature_count, weights_x, weights_y );
    auto weights = memory::allocate(engine, { data_types::f32, weights_format, weights_size });
    auto biases = memory::allocate(engine, { data_types::f32, biases_format, {1,1,output_feature_count,1}});

    //auto output = memory::allocate({output_format, {batch_size, {output_x, output_y}, output_feature_count}});


    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);


    // weights:
    std::vector<float> weights_vals_template {
        -4.0f, -2.0f,
         4.0f,  4.0f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);


    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);


    // output:
    std::vector<float> output_vals_template {
         9.0f, 10.0f,
        13.0f, 14.0f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    // Computing convolution.
    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,stride_x,stride_y },
            { 0,0,0,0 },
            { 1, 1, 1, 1 },
            true,
            0.1f)
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i], get_value<float>(output_ptr, i));
                EXPECT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] << " != " << get_value<float>(output_ptr, i) << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}

TEST(convolution_gpu, basic_yxfb_4_4_yxfb_2_2_b16_if2_of16_st2_2_p0_sp1_fp16)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    engine engine;

    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }


    const auto input_format   = format::yxfb;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = format::bfyx;
#else
    const auto weights_format = format::yxfb;
#endif
    const auto biases_format  = format::bfyx;
    const auto output_format  = input_format;

    const int32_t batch_size = 16;
    const int32_t input_feature_count = 2;
    const int32_t output_feature_count = 16;

    const int32_t stride_x = 2;
    const int32_t stride_y = 2;

    const int32_t input_x = 4;
    const int32_t input_y = 4;
    const int32_t weights_x = 2;
    const int32_t weights_y = 2;
    const int32_t output_x = (input_x - weights_x) / stride_x + 1;
    const int32_t output_y = (input_y - weights_y) / stride_y + 1;


    auto input_size = tensor( batch_size, input_feature_count, input_x, input_y );
    auto input = memory::allocate(engine, { data_types::f32, input_format, input_size });
    auto weights_size = tensor( output_feature_count, input_feature_count, weights_x, weights_y );
    auto weights = memory::allocate(engine, { data_types::f32, weights_format, weights_size });
    auto biases_size = tensor( 1,1,output_feature_count,1 );
    auto biases = memory::allocate(engine, { data_types::f32, biases_format, biases_size });
    auto output_size = tensor( batch_size, output_feature_count, output_x, output_y );
    //auto output = memory::allocate({output_format, {batch_size, {output_x, output_y}, output_feature_count}});

    //auto input_cvtd = memory::allocate(engine, { data_types::f16, input_size });
    //auto weights_cvtd = memory::allocate(engine, { data_types::f16, weights_size });
    //auto biases_cvtd = memory::allocate(engine, { data_types::f16, biases_size });
    //auto output_cvtd  = memory::allocate({output_cvt_format, {batch_size, {output_x, output_y}, output_feature_count}});


    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);


    // weights:
    std::vector<float> weights_vals_template {
        -0.50f, -0.25f,
         0.50f,  0.50f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);


    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);


    // output:
    std::vector<float> output_vals_template {
        1.125f,  1.250f,
        1.625f,  1.750f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    //auto expected_float = memory::allocate(engine, { data_types::f32,{ format::x,{ static_cast<int32_t>(output_vals.size()) } } });
    //auto expected_half  = memory::allocate(engine, { data_types::f16,{ format::x,{ static_cast<int32_t>(output_vals.size()) } } });
    //auto expected       = memory::allocate(engine, { data_types::f32,{ format::x,{ static_cast<int32_t>(output_vals.size()) } } });

//    set_values(expected_float, output_vals);
//    auto cvt_expected_f32_f16 = reorder::create({expected_float, expected_half});
//    auto cvt_expected_f16_f32 = reorder::create({expected_half, expected});
//    execute({cvt_expected_f32_f16, cvt_expected_f16_f32}).wait();
//
//    auto expected_ptr = expected.as<const memory&>().pointer<float>();


    // Computing convolution.
    topology topology(
        input_layout("input", input.get_layout()),
        reorder("cvt_input", "input", {data_types::f16, input_format, input_size}),
        data("weights", weights),
        reorder("cvt_weights", "weights", {data_types::f16, weights_format, weights_size}),
        data("biases", biases),
        reorder("cvt_biases", "biases", {data_types::f16, biases_format, biases_size}),
        convolution(
            "conv",
            "cvt_input",
            { "cvt_weights" },
            { "cvt_biases" },
            { 1,1,stride_x,stride_y }),
        reorder("output", "conv", {data_types::f32, output_format, output_size})
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

#if 0
    for (uint32_t bi = 0; bi < /*batch_size*/ 4; ++bi)
    {
        std::cout << "Actual (batch: " << std::setw(3) << std::right << bi << "):" << std::endl;
        for (uint32_t yi = 0; yi < output_y; ++yi)
        {
            const char* x_sep = "";
            const char* f_sep = "";
            for (uint32_t ofi = 0; ofi < /*output_feature_count*/ 4; ++ofi)
            {
                std::cout << f_sep;
                f_sep = "  |";
                for (uint32_t xi = 0; xi < output_x; ++xi)
                {
                    uint32_t pos = bi + batch_size * (ofi + output_feature_count * (xi + output_x * yi));
                    auto val = get_value(output_ptr, pos);

                    std::cout << x_sep << std::setw(8) << std::setprecision(2) << std::right << std::fixed << val;
                    x_sep = "  ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Expected (batch: " << std::setw(3) << std::right << bi << "):" << std::endl;
        for (uint32_t yi = 0; yi < output_y; ++yi)
        {
            const char* x_sep = "";
            const char* f_sep = "";
            for (uint32_t ofi = 0; ofi < /*output_feature_count*/ 4; ++ofi)
            {
                std::cout << f_sep;
                f_sep = "  |";
                for (uint32_t xi = 0; xi < output_x; ++xi)
                {
                    uint32_t pos = bi + batch_size * (ofi + output_feature_count * (xi + output_x * yi));
                    auto val = output_vals[pos];
                    //auto val = get_value(expected_ptr, pos);

                    std::cout << x_sep << std::setw(8) << std::setprecision(2) << std::right << std::fixed << val;
                    x_sep = "  ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
#endif 

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i] /*get_value(expected_ptr, i)*/, output_ptr[i], 0.002f);
                EXPECT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] /*get_value(expected_ptr, i)*/ << " != " << output_ptr[i] << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}

class convolution_test : public tests::generic_test
{

public:

    static void TearDownTestCase() 
    {
        for (auto generic_params : all_generic_params)
        {
            delete generic_params;
        }

        for (auto layer_params : all_layer_params)
        {
            delete layer_params;
        }
    }

    static std::vector<cldnn::primitive*> generate_specific_test_params()
    {
        // TODO: check split 

        // TODO: check convolution without bias 

        const std::vector<primitive_id>& weights = { "input1" };
        const std::vector<primitive_id>& bias = { "input2" };

        std::vector<tensor> stride_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 2, 3), tensor(1, 1, 4, 1), tensor(1, 1, 5, 5) };
        std::vector<tensor> dilation_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 5, 4), tensor(1, 1, 1, 3), tensor(1, 1, 7, 2) };
        std::vector<tensor> input_offset_sizes = { tensor(0, 0, 0, 0), tensor(0, 0, 2, 2), tensor(0, 0, -5, -2), tensor(0, 0, 3, -3) };

        std::vector<bool> activations = { false, true };
        std::vector<float> activation_slopes = { 0.f, -2.3f };

        // No padding
        all_layer_params.push_back(new convolution("convolution", "input0", weights, bias, stride_sizes[0], input_offset_sizes[0], dilation_sizes[0], activations[0], activation_slopes[0]));
        all_layer_params.push_back(new convolution("convolution", "input0", weights, bias, stride_sizes[1], input_offset_sizes[1], dilation_sizes[1], activations[0], activation_slopes[0]));
        all_layer_params.push_back(new convolution("convolution", "input0", weights, bias, stride_sizes[2], input_offset_sizes[2], dilation_sizes[2], activations[1], activation_slopes[0]));
        all_layer_params.push_back(new convolution("convolution", "input0", weights, bias, stride_sizes[3], input_offset_sizes[3], dilation_sizes[3], activations[1], activation_slopes[1]));

        // Input padding
        all_layer_params.push_back(new convolution("convolution", "reorder0", weights, bias, stride_sizes[1], input_offset_sizes[1], dilation_sizes[1], activations[0], activation_slopes[0]));
        all_layer_params.push_back(new convolution("convolution", "reorder0", weights, bias, stride_sizes[3], input_offset_sizes[3], dilation_sizes[3], activations[1], activation_slopes[1]));

        // Output padding
        all_layer_params.push_back(new convolution("convolution", "input0", weights, bias, stride_sizes[1], input_offset_sizes[1], dilation_sizes[1], activations[0], activation_slopes[0], { { 0, 0, 2, 4 },{ 0, 0, 0, 19 } }));
        all_layer_params.push_back(new convolution("convolution", "input0", weights, bias, stride_sizes[2], input_offset_sizes[2], dilation_sizes[2], activations[1], activation_slopes[0], { { 0, 0, 1, 0 },{ 0, 0, 13, 9 } }));

        // Input + Output padding
        all_layer_params.push_back(new convolution("convolution", "reorder0", weights, bias, stride_sizes[0], input_offset_sizes[0], dilation_sizes[0], activations[0], activation_slopes[0], { { 0, 0, 1, 5 },{ 0, 0, 19, 4 } }));
        all_layer_params.push_back(new convolution("convolution", "reorder0", weights, bias, stride_sizes[3], input_offset_sizes[3], dilation_sizes[3], activations[1], activation_slopes[1], { { 0, 0, 1, 2 },{ 0, 0, 3, 4 } }));

        return all_layer_params;
    }

    static std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> generate_all_test_params()
    {
        generate_specific_test_params();

        std::vector<cldnn::format> input_formats = { cldnn::format::bfyx, cldnn::format::yxfb };
        std::vector<cldnn::format> weights_formats = { cldnn::format::bfyx, cldnn::format::yxfb };

        std::vector<int32_t> output_features_sizes = { 1, 3, 16 };
        std::vector<cldnn::tensor> kernel_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 4, 7), tensor(1, 1, 5, 3) };

        std::vector<tensor> input_tensor_size = { tensor(1, 5, 59, 72), tensor(8, 3, 63, 56), tensor(16, 2, 50, 50), tensor(32, 1, 44, 62) };

        for (cldnn::data_types data_type : test_data_types)
        {
            for (cldnn::format input_format : input_formats)
            {
                for (cldnn::format weights_format : weights_formats)
                {
                    if ((input_format == cldnn::format::yxfb) && (weights_format == cldnn::format::bfyx) && (data_type == cldnn::data_types::f16))
                    {
                        // yxfb oiyx convolution not supported in FP16.
                        continue;
                    }
                    cldnn::build_options network_build_options;
                    if (input_format == cldnn::format::bfyx)
                    {
                        network_build_options.set_option(cldnn::build_option::optimize_data(true));
                    }
                    for (cldnn::tensor input_size : input_tensor_size)
                    {
                        for (cldnn::tensor kernel_size : kernel_sizes)
                        {
                            for (auto output_features : output_features_sizes)
                            {
                                test_params* params = new test_params(data_type, input_format, input_size.batch[0], input_size.feature[0], tensor(1, 1, input_size.spatial[0], input_size.spatial[1]), network_build_options);
                                int input_features = params->input_layouts[0].size.feature[0];
                                params->input_layouts.push_back(cldnn::layout(params->data_type, weights_format, cldnn::tensor(output_features, input_features, kernel_size.spatial[0], kernel_size.spatial[1]))); // weights
                                params->input_layouts.push_back(cldnn::layout(params->data_type, params->fmt, cldnn::tensor(1, 1, output_features, 1))); // biases
                                all_generic_params.push_back(params);
                            }
                        }
                    }
                }
            }
        }

        // Create all the combinations for the test.
        for (cldnn::primitive* layer_param : all_layer_params)
        {
            for (tests::test_params* test_param : all_generic_params)
            {
                const cldnn::convolution* convolution = (cldnn::convolution*)layer_param;
                if ((test_param->fmt == cldnn::format::yxfb) && (convolution->input[0] == "reorder0"))
                {
                    // Input padding not support in yxfb convolution.
                    continue;
                }
                all_test_params.push_back(std::make_tuple(test_param, layer_param));
            }
        }

        return all_test_params;
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        return ((format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_yxfb));
    }

    virtual cldnn::tensor get_expected_output_tensor()
    {
        const cldnn::convolution* convolution = (cldnn::convolution*)layer_params;
        tensor input_size = generic_params->input_layouts[0].size;
        tensor dilation = convolution->dilation;
        tensor stride = convolution->stride;
        tensor input_offset = convolution->input_offset;
        tensor weights_size = generic_params->input_layouts[1].size;

        int kernel_extent_y = dilation.spatial[1] * (weights_size.spatial[1] - 1) + 1;
        int kernel_extent_x = dilation.spatial[0] * (weights_size.spatial[0] - 1) + 1;

        // Calculate output size
        int output_size_y = 1 + (input_size.spatial[1] - kernel_extent_y - 2 * input_offset.spatial[1]) / stride.spatial[1];
        int output_size_x = 1 + (input_size.spatial[0] - kernel_extent_x - 2 * input_offset.spatial[0]) / stride.spatial[0];
        int output_features = weights_size.batch[0];

        return cldnn::tensor(input_size.batch[0], output_features, output_size_x, output_size_y);
    }

    virtual void prepare_input_for_test(std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            prepare_input_for_test_typed<float>(inputs);
        }
        else
        {
            prepare_input_for_test_typed<FLOAT16>(inputs);
        }
    }

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory>& inputs)
    {
        int k = (generic_params->data_type == data_types::f32) ? 8 : 4;

        // Update inputs.
        auto input = inputs[0];
        auto input_size = inputs[0].get_layout().size;
        VVVVF<Type> input_rnd = generate_random_4d<Type>(input_size.batch[0], input_size.feature[0], input_size.spatial[1], input_size.spatial[0], -2, 2, k);
        VF<Type> input_rnd_vec = flatten_4d<Type>(input.get_layout().format, input_rnd);
        set_values(input, input_rnd_vec);

        // Update weights.
        auto weight_input = inputs[1];
        auto weight_size = inputs[1].get_layout().size;
        VVVVF<Type> weight_rnd = generate_random_4d<Type>(weight_size.batch[0], weight_size.feature[0], weight_size.spatial[1], weight_size.spatial[0], -2, 2, k);
        VF<Type> weight_rnd_vec = flatten_4d<Type>(weight_input.get_layout().format, weight_rnd);
        set_values(weight_input, weight_rnd_vec);

        // Update biases.
        auto bias_input = inputs[2];
        auto bias_size = inputs[2].get_layout().size;
        VF<Type> bias_rnd = generate_random_1d<Type>(bias_size.spatial[0], -2, 2, k);
        set_values(bias_input, bias_rnd);
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        // Output reference is always bfyx.

        const cldnn::convolution* convolution = (cldnn::convolution*)layer_params;

        data_types dt = inputs[0].get_layout().data_type;
        
        tensor input_size = inputs[0].get_layout().size;
        tensor dilation = convolution->dilation;
        tensor stride = convolution->stride;
        bool is_relu_fused = convolution->with_activation;
        float activation_slope = convolution->activation_negative_slope;
        tensor input_offset = convolution->input_offset;
        tensor weights_size = inputs[1].get_layout().size;
        padding output_padding = convolution->output_padding;

        tensor output_size = get_expected_output_tensor();

        // Calculate output size
        int output_size_y = output_size.spatial[1];
        int output_size_x = output_size.spatial[0];
        int output_features = weights_size.batch[0];
        int input_features = weights_size.feature[0];

        auto output = memory::allocate( engine, cldnn::layout(dt, cldnn::format::bfyx, output_size, output_padding) );

        auto input_mem = inputs[0].pointer<Type>();
        auto weights_mem = inputs[1].pointer<Type>();
        auto bias_mem = inputs[2].pointer<Type>();
        auto output_mem = output.pointer<Type>();

        tensor output_buffer_size = output.get_layout().get_buffer_size();

        // Initialized output with zeros.
        std::fill(output_mem.begin(), output_mem.end(), static_cast<Type>(0));
    
        // Add the bias
        for (int b = 0; b < input_size.batch[0]; b++)
        {
            for (int out_f = 0; out_f < output_features; out_f++)
            {
                for (int y = 0; y < output_size_y; y++)
                {
                    for (int x = 0; x < output_size_x; x++)
                    {
                        int output_index = (b * output_buffer_size.feature[0] + out_f) * output_buffer_size.spatial[1] * output_buffer_size.spatial[0];
                        tensor lower_output_padding = convolution->output_padding.lower_size();
                        output_index += (lower_output_padding.spatial[1] + y) * output_buffer_size.spatial[0] + lower_output_padding.spatial[0] + x;

                        output_mem[output_index] += bias_mem[out_f];
                    }
                }
            }
        }

        const auto input0_desc = get_linear_memory_desc(inputs[0].get_layout());
        const auto input1_desc = get_linear_memory_desc(inputs[1].get_layout());

        // Convolve with weights
        for (int b = 0; b < input_size.batch[0]; b++)
        {
            int input_bi = b;
            for (int out_f = 0; out_f < output_features; out_f++)
            {
                for (int in_f = 0; in_f < input_features; in_f++)
                {
                    int input_fi = in_f;
                    for (int y = 0; y < output_size_y; y++)
                    {
                        for (int x = 0; x < output_size_x; x++)
                        {
                            int output_bi = b;
                            int output_fi = out_f;
                            int output_yi = y;
                            int output_xi = x;
                            int output_index = (output_bi * output_buffer_size.feature[0] + output_fi) * output_buffer_size.spatial[1] * output_buffer_size.spatial[0];
                            tensor lower_output_padding = convolution->output_padding.lower_size();
                            output_index += (lower_output_padding.spatial[1] + output_yi) * output_buffer_size.spatial[0] + lower_output_padding.spatial[0] + output_xi;

                            for (int kernel_y = 0; kernel_y < weights_size.spatial[1]; kernel_y++)
                            {
                                int input_yi = y * stride.spatial[1] + input_offset.spatial[1] + kernel_y * dilation.spatial[1];
                                if ((input_yi < 0) || (input_yi >= input_size.spatial[1]))
                                {
                                    continue;
                                }

                                for (int kernel_x = 0; kernel_x < weights_size.spatial[0]; kernel_x++)
                                {
                                    int input_xi = x * stride.spatial[0] + input_offset.spatial[0] + kernel_x * dilation.spatial[0];
                                    if ((input_xi < 0) || (input_xi >= input_size.spatial[0]))
                                    {
                                        continue;
                                    }

                                    size_t input_index = get_linear_index(inputs[0].get_layout(), input_bi, input_fi, input_yi, input_xi, input0_desc);

                                    int weight_bi = out_f;
                                    int weight_fi = in_f;
                                    int weight_yi = kernel_y;
                                    int weight_xi = kernel_x;
                                    size_t weight_index = get_linear_index(inputs[1].get_layout(), weight_bi, weight_fi, weight_yi, weight_xi, input1_desc);
                                    output_mem[output_index] += input_mem[input_index] * weights_mem[weight_index];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Relu activation
        if (is_relu_fused)
        {
            for (int i = 0; i < (int)output_buffer_size.count(); i++)
            {
                output_mem[i] = (output_mem[i] > 0.f) ? output_mem[i] : (output_mem[i] * (Type)activation_slope);
            }
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;
    static std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> all_test_params;
};

std::vector<tests::test_params*> convolution_test::all_generic_params = {};
std::vector<cldnn::primitive*> convolution_test::all_layer_params = {};
std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> convolution_test::all_test_params = {};

TEST_P(convolution_test, DISABLED_test_all)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(CONVOLUTION, 
                        convolution_test, 
                        ::testing::ValuesIn(convolution_test::generate_all_test_params()),
                        tests::generic_test::custom_param_name_functor());
