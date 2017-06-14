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
#include <algorithm>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/activation.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include "api/CPP/reorder.hpp"

namespace{
    auto calc_idx = [](std::vector<uint32_t> yxfb_pos, std::vector<uint32_t>& buf_size_bfyx) -> uint32_t{
        return yxfb_pos[3]
             + yxfb_pos[2] * buf_size_bfyx[0]
             + yxfb_pos[1] * buf_size_bfyx[0] * buf_size_bfyx[1]
             + yxfb_pos[0] * buf_size_bfyx[0] * buf_size_bfyx[1] * buf_size_bfyx[2];
    };
}

namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;

template <typename T>
VVVVF<T> relu_reference(VVVVF<T> &input, T slope = 0.0f,
    int input_padding_y = 0, int input_padding_x = 0,
    int output_padding_y = 0, int output_padding_x = 0) {

    size_t padding_y = input_padding_y + output_padding_y;
    size_t padding_x = input_padding_x + output_padding_x;
    size_t output_b = input.size();
    size_t output_f = input[0].size();
    size_t output_y = input[0][0].size() + 2 * padding_y;
    size_t output_x = input[0][0][0].size() + 2 * padding_x;
    VVVVF<T> output(output_b, VVVF<T>(output_f, VVF<T>(output_y, VF<T>(output_x))));

    for (size_t b = 0; b < output_b; ++b) {
        for (size_t f = 0; f < output_f; ++f) {
            for (size_t y = 0; y < input[0][0].size(); ++y) {
                for (size_t x = 0; x < input[0][0][0].size(); ++x) {
                    output[b][f][y + padding_y][x + padding_x] = input[b][f][y][x];
                    if (input[b][f][y][x] < (T)0)
                        output[b][f][y + padding_y][x + padding_x] *= slope;
                }
            }
        }
    }
    return output;
}

template <typename T>
void generic_relu_test(cldnn::format test_input_fmt, int input_b, int input_f, int input_y, int input_x, T slope,
    int input_padding_y, int input_padding_x, int output_padding_y, int output_padding_x) {

    int min_random = -2, max_random = 2;
    VVVVF<T> input_rnd = generate_random_4d<T>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
    
    engine engine;
    tensor input_tensor( input_b, input_f, input_x, input_y );
    auto input = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
    set_values(input, input_rnd_vec);
    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, input_padding_x, input_padding_y }, 0 })),
        activation(
            "relu",
            "reorder",
            slope,
            { { 0, 0, output_padding_x, output_padding_y }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<T>();

    EXPECT_EQ(output_layout.format.value, test_input_fmt.value);
    tensor output_tensor = output_layout.size;
    int y_size = output_tensor.spatial[1];
    int x_size = output_tensor.spatial[0];
    int f_size = output_tensor.feature[0];
    int b_size = output_tensor.batch[0];
    EXPECT_EQ(y_size, input_y + 2 * output_padding_y);
    EXPECT_EQ(x_size, input_x + 2 * output_padding_x);
    EXPECT_EQ(f_size, input_f);
    EXPECT_EQ(b_size, input_b);
    
    bool test_is_correct = true;
    VVVVF<T> output_cpu = relu_reference<T>(input_rnd, slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
    VF<T> output_cpu_vec = flatten_4d<T>(test_input_fmt, output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (!floating_point_equal(output_cpu_vec[i], output_ptr[i])) {
            test_is_correct = false;
            break;
        }
    }
    EXPECT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "input_b = " << input_b << std::endl
        << "input_f = " << input_f << std::endl
        << "input_y = " << input_y << std::endl
        << "input_x = " << input_x << std::endl
        << "slope = " << (float)slope << std::endl
        << "input_padding_y = " << input_padding_y << std::endl
        << "input_padding_x = " << input_padding_x << std::endl
        << "output_padding_y = " << output_padding_y << std::endl
        << "output_padding_x = " << output_padding_x << std::endl
        << "type = " << (sizeof(T) == 2 ? "float16" : "float32") << std::endl;
}

TEST(relu_f32_fw_gpu, basic_yxfb) {
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
      2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
      3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
      1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
        3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -0.5f, 1.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation( "relu", "input", 0.5, { { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(relu_f32_fw_gpu, basic_input_padding_yxfb) {
    //  Input Padding: 2x1 (yx format) out of the reorder layer
    //  The expected size is the same as in put - the output padding is set to 0, 0
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1 -2 -3  4  5  z
    //  z  2  2  3  4 -6  z
    //  z  3 -3  3  5  1  z
    //  z  1  1  1 -1  1  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Slope: 0.5
    //
    //  Output:
    //  1   -1   -1.5  4    5
    //  2    2    3    4   -3
    //  3   -1.5  3    5    1
    //  1    1    1   -0.5  1

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });

    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
         1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
         2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
         3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
         1.0f, 1.0f, 1.0f, -0.5f, 1.0f};

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("reorder", "input", input.get_layout().with_padding({ { 0, 0, 2, 1 }, 0 })),
        activation("relu", "reorder", 0.5, { { 0, 0, 0, 0 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(relu_f32_fw_gpu, basic_output_padding_yxfb) {
    //  Output Padding: 3x3 (yx format)
    //
    //  Input:
    //  1 -2 -3  4  5
    //  2  2  3  4 -6
    //  3 -3  3  5  1
    //  1  1  1 -1  1
    //
    //  Slope: 0.5
    //
    //  Output:
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    1   -1   -1.5  4    5    0    0    0
    //  0    0    0    2    2    3    4   -3    0    0    0
    //  0    0    0    3   -1.5  3    5    1    0    0    0
    //  0    0    0    1    1    1   -0.5  1    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0
    //  0    0    0    0    0    0    0    0    0    0    0

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    set_values(input,
    { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
        2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
        3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
        1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
    VF<float> output_vec = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.5f, 4.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 3.0f, 4.0f, -3.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 3.0f, -1.5f, 3.0f, 5.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    topology topology(
        input_layout("input", input.get_layout()),
        activation("relu", "input", 0.5, { { 0, 0, 3, 3 }, 0 }));
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu");

    auto output_memory = outputs.at("relu").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_size = output_layout.get_buffer_size();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_size.spatial[1];
    int x_size = output_size.spatial[0];
    int f_size = output_size.feature[0];
    int b_size = output_size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 10);
    EXPECT_EQ(x_size, 11);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(output_vec[i], output_ptr[i]);
    }
}

TEST(DISABLED_relu_gpu, generic_random) {
    VF<cldnn::format> test_inputs_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
    VF<float> slopes = { 0.0f, -0.0f, -17.19f, 1028.8f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() };
    std::vector<std::pair<int, int>> input_sizes = { { 100, 100 },{ 227, 227 },{ 400, 600 },{ 531, 777 },{ 4096, 1980 } };
    
    engine engine;
    bool f16_supported = !!engine.get_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    for (cldnn::format test_input_fmt : test_inputs_fmts) {
        for (int input_b = 1; input_b <= 16; input_b *= 2) {
            for (int input_f = 1; input_f <= 1; ++input_f) {
                for (std::pair<int, int> &input_yx : input_sizes) {
                    for (float slope : slopes) {
                        for (int input_padding_y = 0; input_padding_y <= 0; ++input_padding_y) {
                            for (int input_padding_x = 0; input_padding_x <= 0; ++input_padding_x) {
                                for (int output_padding_y = 0; output_padding_y <= 1; ++output_padding_y) {
                                    for (int output_padding_x = 0; output_padding_x <= 1; ++output_padding_x) {
                                        generic_relu_test<float>(test_input_fmt, input_b, input_f, input_yx.first, input_yx.second, slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
                                        if (!f16_supported) continue;
                                        generic_relu_test<FLOAT16>(test_input_fmt, input_b, input_f, input_yx.first, input_yx.second, (FLOAT16)slope, input_padding_y, input_padding_x, output_padding_y, output_padding_x);
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

#if 0 // no support yet
TEST(relu_f32_fw_gpu, basic) {
    // FAIL now, because we don't support using the same buffer as input and output
    EXPECT_EQ(false, true);
    return;

    const uint32_t y = 8, x = 8, f = 3, b = 2;

    auto input = memory::allocate({  memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto output = memory::describe({  memory::format::yxfb_f32,{ b,{ y, x }, f } });
    fill<float>(input.as<const memory&>());

    auto act = relu::create({  output, input });
    auto buf = input.as<const memory&>().pointer<float>();
    // write output to input buffer
    execute({ output(buf), act }).wait();

    // multiply all positive intigers by -1
    for (size_t i = 0; i < y*x*f*b; ++i)
        buf[i] = (buf[i] > 0) ? -buf[i] : buf[i];

    execute({ act }).wait();

    bool result = false;
    // every element should be 0.0f
    for (size_t i = 0; i < y*x*f*b; ++i)
        result = result || buf[i];

    EXPECT_EQ(false, result);
}
#endif // NOT YET

#if 0 
- rewrite this
TEST(relu_f32_fw_gpu, intrinsics_avx2) {
    const uint32_t y = 8, x = 8, f = 3, b = 2;

    // Optimized data
    auto input = memory::allocate({  memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto output = memory::allocate({  memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto& input_memory = input.as<const memory&>();
    auto& output_memory = output.as<const memory&>();

    // Reference data
    auto ref_output = memory::allocate({  memory::format::yxfb_f32,{ b,{ y, x }, f } });
    auto& ref_output_memory = ref_output.as<const memory&>();

    // Initialize input data
    fill<float>(input_memory);

    // Relu primitives
    auto opt_relu = relu::create({ engine::reference, output, input });
    auto ref_relu = relu::create({  ref_output, input });

    execute({ output, opt_relu }).wait();
    execute({ ref_output, ref_relu }).wait();

    {
        auto ref_out_ptr = ref_output_memory.pointer<float>();
        auto out_ptr = output_memory.pointer<float>();
        for (size_t output_element = 0; output_element < output_memory.count(); ++output_element)
            EXPECT_EQ(true, tests::are_equal(ref_out_ptr[output_element], out_ptr[output_element]));
    }
}
#endif



class relu_test: public tests::generic_test
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
        // No padding
        all_layer_params.push_back(new activation("relu", "input0", 0));
        all_layer_params.push_back(new activation("relu", "input0", -17.19f));
        all_layer_params.push_back(new activation("relu", "input0", 1028.8f));
        all_layer_params.push_back(new activation("relu", "input0", std::numeric_limits<float>::infinity()));

        // Output padding
        all_layer_params.push_back(new activation("relu", "input0", -5.4f, { { 0, 0, 11, 5 },{ 0, 0, 3, 19 } }));
        all_layer_params.push_back(new activation("relu", "input0", -1.f, { { 0, 0, 5, 13 },{ 0, 0, 1, 2 } }));

        // Input padding (output of reorder layer)
        all_layer_params.push_back(new activation("relu", "reorder0", -2.3f));
        all_layer_params.push_back(new activation("relu", "reorder0", 23.4f));

        // Input padding (output of reorder layer) + Output padding
        all_layer_params.push_back(new activation("relu", "reorder0", 1.003f, { { 0, 0, 1, 2 },{ 0, 0, 3, 4 } }));
        all_layer_params.push_back(new activation("relu", "reorder0", -2.05f, { { 0, 0, 2, 0 },{ 0, 0, 5, 9 } }));

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params()
    {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        return ((format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_yxfb));
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        const cldnn::activation* relu = (cldnn::activation*)layer_params;

        //Output is bfyx
        data_types dt = inputs[0].get_layout().data_type;
        auto output = memory::allocate(engine, cldnn::layout(dt, cldnn::format::bfyx, inputs[0].get_layout().size, relu->output_padding));

        float negative_slope = relu->negative_slope;

        auto input_mem = inputs[0].pointer<Type>();
        auto output_mem = output.pointer<Type>();

        int batch = inputs[0].get_layout().size.batch[0];
        int feature = inputs[0].get_layout().size.feature[0];
        int height = inputs[0].get_layout().size.spatial[1];
        int width = inputs[0].get_layout().size.spatial[0];

        int output_height = output.get_layout().get_buffer_size().spatial[1];
        int output_width = output.get_layout().get_buffer_size().spatial[0];

        for (int b = 0; b < batch; ++b)
        {
            for (int f = 0; f < feature; ++f)
            {
                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        size_t input_index = get_linear_index(inputs[0].get_layout(), b, f, y, x);

                        int output_index = (b * feature + f) * output_height * output_width;
                        tensor lower_padding = relu->output_padding.lower_size();
                        output_index += (lower_padding.spatial[1] + y) * output_width + lower_padding.spatial[0] + x;

                        output_mem[output_index] = (input_mem[input_index] >= (Type)0.f) ? input_mem[input_index] : input_mem[input_index] * (Type)negative_slope;
                    }
                }
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

};

std::vector<tests::test_params*> relu_test::all_generic_params = {};
std::vector<cldnn::primitive*> relu_test::all_layer_params = {};

TEST_P(relu_test, DISABLED_test_all)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(RELU, 
                        relu_test, 
                        ::testing::Combine(::testing::ValuesIn(relu_test::generate_generic_test_params()),
                                           ::testing::ValuesIn(relu_test::generate_specific_test_params())),
                        tests::generic_test::custom_param_name_functor());


class prelu_test : public tests::generic_test
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
        // No padding
        all_layer_params.push_back(new activation("relu", "input0", "input1"));

        // Output padding
        all_layer_params.push_back(new activation("relu", "input0", "input1", { { 0, 0, 11, 5 },{ 0, 0, 3, 19 } }));
        all_layer_params.push_back(new activation("relu", "input0", "input1", { { 0, 0, 5, 13 },{ 0, 0, 1, 2 } }));

        // Input padding (output of reorder layer)
        all_layer_params.push_back(new activation("relu", "reorder0", "input1"));

        // Input padding (output of reorder layer) + Output padding
        all_layer_params.push_back(new activation("relu", "reorder0", "input1", { { 0, 0, 1, 2 },{ 0, 0, 3, 4 } }));
        all_layer_params.push_back(new activation("relu", "reorder0", "input1", { { 0, 0, 2, 0 },{ 0, 0, 5, 9 } }));

        return all_layer_params;
    }

    static std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> generate_all_test_params()
    {
        generic_test::generate_generic_test_params(all_generic_params);
        generate_specific_test_params();

        // Prepare slope input for prelu layer (size should be equal to input feature size - one slope per feature).
        for (tests::test_params* test_param : all_generic_params)
        {
            cldnn::tensor input_size = test_param->input_layouts[0].size;
            test_param->input_layouts.push_back(cldnn::layout(test_param->data_type, test_param->fmt, cldnn::tensor(1, 1, input_size.feature[0], 1)));
        }

        // Create all the combinations for the test.
        for (cldnn::primitive* layer_param : all_layer_params)
        {
            for (tests::test_params* test_param : all_generic_params)
            {
                all_test_params.push_back(std::make_tuple(test_param, layer_param));
            }
        }

        return all_test_params;
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        return ((format == cldnn_format_type::cldnn_format_bfyx) || (format == cldnn_format_type::cldnn_format_yxfb));
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        const cldnn::activation* relu = (cldnn::activation*)layer_params;

        //Output is bfyx
        data_types dt = inputs[0].get_layout().data_type;
        auto output = memory::allocate(engine, cldnn::layout(dt, cldnn::format::bfyx, inputs[0].get_layout().size, relu->output_padding));

        auto input_mem = inputs[0].pointer<Type>();
        auto slope_mem = inputs[1].pointer<Type>();
        auto output_mem = output.pointer<Type>();

        int batch = inputs[0].get_layout().size.batch[0];
        int feature = inputs[0].get_layout().size.feature[0];
        int height = inputs[0].get_layout().size.spatial[1];
        int width = inputs[0].get_layout().size.spatial[0];

        assert((int)inputs[1].get_layout().count() == feature);

        int output_height = output.get_layout().get_buffer_size().spatial[1];
        int output_width = output.get_layout().get_buffer_size().spatial[0];

        for (int b = 0; b < batch; ++b)
        {
            for (int f = 0; f < feature; ++f)
            {
                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        size_t input_index = get_linear_index(inputs[0].get_layout(), b, f, y, x);

                        int output_index = (b * feature + f) * output_height * output_width;
                        tensor lower_padding = relu->output_padding.lower_size();
                        output_index += (lower_padding.spatial[1] + y) * output_width + lower_padding.spatial[0] + x;

                        output_mem[output_index] = (input_mem[input_index] >= (Type)0.f) ? input_mem[input_index] : input_mem[input_index] * slope_mem[f];
                    }
                }
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

std::vector<tests::test_params*> prelu_test::all_generic_params = {};
std::vector<cldnn::primitive*> prelu_test::all_layer_params = {};
std::vector<std::tuple<tests::test_params*, cldnn::primitive*>> prelu_test::all_test_params = {};

TEST_P(prelu_test, DISABLED_test_all)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(PRELU, 
                        prelu_test, 
                        ::testing::ValuesIn(prelu_test::generate_all_test_params()),
                        tests::generic_test::custom_param_name_functor());
