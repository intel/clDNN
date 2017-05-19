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

#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/softmax.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace std;
using namespace tests;


class softmax_gpu_xb_f32_test_fixture: public ::testing::Test {
public:
    static const int32_t
        output_x  = 10, output_b  = 2,  // size of whole output buffer
        input_x   = 10, input_b   = 2,  // size of whole input buffer
        in_size   = input_x*input_b,
        out_size  = output_x*output_b;


    float in_buffer[in_size];
    float out_buffer[out_size];
    float expected_buffer[out_size];

    cldnn::engine engine;
    cldnn::memory input;
    //neural::primitive output = memory::allocate({ memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    softmax_gpu_xb_f32_test_fixture()
        :engine()
        ,input(memory::allocate(engine, { data_types::f32, format::yxfb, { input_b, 1, input_x, 1}}))
    {}

    void compare_out_buffer_with_expected() {
        for(size_t i = 0; i < out_size; ++i) {
            // does output have expected values
            EXPECT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    }

    void compare_out_buffer_with_expected_batch_wise() {
        for(size_t b = 0; b < output_b; ++b) {
            float batch_wise_sum = 0;
            for(size_t x = 0; x < output_x; ++x) {
                auto idx = b+x*output_b;
                batch_wise_sum += out_buffer[idx];
                // does output have expected values
                EXPECT_TRUE(are_equal(out_buffer[idx], expected_buffer[idx]))
                    << "At ["<< idx <<  "] Expected : " << expected_buffer[idx] << " actual : " << out_buffer[idx];
            }
            // does it sum to 1 batch wise
            EXPECT_TRUE(are_equal(batch_wise_sum, 1.0f))
                << "Expected : " << 1.0f << " actual : " << batch_wise_sum;
        }
    }
};

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values) {
// in_buffer filled with same value == 1.0f
    for(uint32_t i = 0; i < out_size; ++i) {
              in_buffer[i] = 1.0f;
        expected_buffer[i] = 0.1f;
    }
    std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));

    set_values(input, in_b);

    network network(engine, topology(input_layout("input", input.get_layout()), softmax("softmax", "input")));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    compare_out_buffer_with_expected();
}

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_batch_wise) {
// in_buffer filled with same value == 1..2 each batch accordingly (softmax can only xb_f32 )
    for(size_t i = 0; i < output_x; ++i) {
        for(size_t j = 0; j < output_b; ++j)
            in_buffer[j+i*output_b] = (j+i*output_b) % 2 +1.0f;
    }

    std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));
    set_values(input, in_b);
    // fill buffer with the expected 0.1f value
    for(size_t i = 0; i < out_size; ++i)
        expected_buffer[i] = 0.1f;

    network network(engine, topology(input_layout("input", input.get_layout()), softmax("softmax", "input")));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    compare_out_buffer_with_expected_batch_wise();
}

TEST_F(softmax_gpu_xb_f32_test_fixture, values_batch_wise) {

    float in_buf[in_size] = {
       //b0  b1
        2.0f, 2.0f, //x0
        2.0f, 2.0f, //x1
        2.0f, 2.0f, //x2
        3.0f, 3.0f, //x3
        5.0f, 5.0f, //x4
        4.0f, 4.0f, //x5
        3.0f, 3.0f, //x6
        2.0f, 2.0f, //x7
        2.0f, 2.0f, //x8
        2.0f, 2.0f  //x9
    };

    float exp_buf[out_size] = {
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f,
        0.069858674f,    0.069858674f,
        0.516189665f,    0.516189665f,
        0.189895565f,    0.189895565f,
        0.069858674f,    0.069858674f,
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f,
        0.02569957f,	 0.02569957f

    };

    std::vector<float> in_b(std::begin(in_buf), std::end(in_buf));
    set_values(input, in_b);
    std::copy(exp_buf, exp_buf+in_size, expected_buffer);

    // out_buffer filled with non-signaling NaN
    for(size_t i = 0; i < out_size; ++i)
        out_buffer[i] = NAN;

    network network(engine, topology(input_layout("input", input.get_layout()), softmax("softmax", "input")));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    compare_out_buffer_with_expected_batch_wise();
}

TEST(softmax_gpu_bfyx_f32, sum_to_one_per_feature) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(softmax("softmax", "input"));

    set_values(input, {  //bfyx    
             //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    });

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    auto output_ptr = output.pointer<float>();
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    float sum = 0;
    float expected_sum = 1.0f;
    
    for (uint32_t i = 0; i < batch_num; i++) //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
    {
        for (uint32_t j = 0; j < y_size; j++)
        {
            for (uint32_t k = 0; k < x_size; k++)
            {
                for (uint32_t l = 0; l < feature_num; l++)
                {
                    int index = i * feature_num * x_size * y_size + j * x_size + k + l * x_size * y_size;
                    sum += out_buffer[index];
                }
                EXPECT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }

    }
}

TEST(softmax_gpu_bfyx_f32, check_max_values_corectness) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(softmax("softmax", "input"));

    vector<float> input_vec = {
               //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253f, 0.364855f, 0.9999f, 0.493895f,
        0.719295f, 0.364855f, 0.731059f, 0.977054f
    };

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    auto output_ptr = output.pointer<float>();
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }


    float temp_max = 0;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
    {
        for (uint32_t j = 0; j < y_size; j++)
        {
            for (uint32_t k = 0; k < x_size; k++)
            {
                for (uint32_t l = 0; l < feature_num; l++)
                {
                    int index = i * feature_num * x_size * y_size + j * x_size + k + l * x_size * y_size;
                    if (out_buffer[index] >= temp_max)
                    {
                        temp_max = out_buffer[index];
                    }       
                }
                EXPECT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;
            }
        }
    }
}
//TEST(softmax_gpu_xb_f32_test, basic_with_offsets) {
//
//    const uint32_t output_x  = 7, output_b  = 3,  // size of whole output buffer
//                   input_x   = 6, input_b   = 2,  // size of whole input buffer
//                   out_off_x = 0, out_off_b = 1,
//                   out_siz_x = 5, out_siz_b = 2;  // size of area to do softmax after offset
//
//    const int32_t  in_off_x  = 1, in_off_b  = 0;
//
//    float in_buffer[input_x*input_b];
//    float out_buffer[output_x*output_b];
//    // input buffer should be initialized with valid data
//
//    auto input  = memory::allocate({ memory::format::xb_f32, {input_b, {{input_x}}, 1}});
//    auto output = memory::allocate({ memory::format::xb_f32, {output_b, {{output_x}}, 1}});
//
//    auto act    = normalization::softmax::create({
//                                                  output,
//                                                  {out_off_b, {{out_off_x}}, 0},
//                                                  {out_siz_b, {{out_siz_x}}, 1},
//                                                  input,
//                                                  {in_off_b, {{in_off_x}}, 0}
//                                                 });
//    // in_buffer filled with same value == 1.0f
//    for(size_t i = 0; i < input_x*input_b; ++i)
//        in_buffer[i] = 1.0f;
//
//    std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));
//    set_values(input, in_b);
//
//    const float out_of_offset_value = NAN;
//    // out_buffer filled with non-signaling NaN
//    for(size_t i = 0; i < output_x*output_b; ++i)
//        out_buffer[i] = out_of_offset_value;
//
//    std::vector<float> out_b(std::begin(out_buffer), std::end(out_buffer));
//    set_values(output, out_b);
//
//    execute({input, output, act}).wait();
//
//    auto& output_memory = output.as<const memory&>();
//    for (int i = 0; i < output_x*output_b; i++)
//    {
//        out_buffer[i] = get_value<float>(output_memory, i);
//    }
//
//    auto expected_value = 0.2f;
//    auto end_b = out_off_b+out_siz_b;
//    auto end_x = out_off_x+out_siz_x;
//
//    for(size_t x = 0; x < output_x; ++x)
//        for(size_t b = 0; b < output_b; ++b) {
//            auto idx = b+x*output_b;
//            float value = out_buffer[idx];
//            float expected = (b >= out_off_b && b < end_b) && (x >= out_off_x && x < end_x) //is in range ?
//                ? expected_value       // valid value that's in data range
//                : out_of_offset_value; // invalid value (non-signaling NaN) for skipped buffer positions (bof offsets)
//          EXPECT_TRUE(are_equal(value, expected))
//              << "At ["<< idx <<  "] Expected : " << expected << " actual :" << value;
//        }
//};
//
//TEST(softmax_xb_f32_fw, intrinsics_avx2_batch1_sum_to_one) {
//    const uint32_t x = 1000, b = 1;
//
//    auto input  = memory::allocate({memory::format::xb_f32, {b, {x}}});
//    auto output = memory::allocate({memory::format::xb_f32, {b, {x}}});
//    auto& input_memory  = input.as<const memory&>();
//    auto output_memory_ptr = output.as<const memory&>().pointer<float>();
//
//    // Initialize input data
//    fill<float>(input_memory);
//
//    auto softmax = normalization::softmax::create({output, input});
//
//    execute({output, softmax}).wait();
//
//    auto sum = accumulate(output_memory_ptr, output_memory_ptr + x, 0.0f);
//
//    EXPECT_EQ(true, tests::are_equal(sum, 1.0f));
//}
//
//TEST(softmax_xb_f32_fw, intrinsics_avx2_batch1_ref_compare) {
//    const uint32_t x = 100, b = 1;
//
//    // Optimized data
//    auto input  = memory::allocate({ memory::format::xb_f32, {b, {x}} });
//    auto output = memory::allocate({ memory::format::xb_f32, {b, {x}} });
//    auto& input_memory  = input.as<const memory&>();
//    auto& output_memory = output.as<const memory&>();
//
//    // Reference data
//    auto ref_output = memory::allocate({ memory::format::xb_f32,{b, {x}} });
//    auto& ref_output_memory = ref_output.as<const memory&>();
//
//    // Initialize input data
//    fill<float>(input_memory);
//
//    // Softmax primitives
//    auto opt_softmax = normalization::softmax::create({ output, input });
//    auto ref_softmax = normalization::softmax::create({ ref_output, input });
//
//    execute({output, opt_softmax}).wait();
//    execute({ref_output, ref_softmax}).wait();
//
//    for(uint32_t output_element = 0; output_element < output_memory.count(); ++output_element)
//        EXPECT_EQ(true, tests::are_equal(get_value<float>(ref_output_memory, output_element), get_value<float>(output_memory, output_element)));
//}
//
//TEST(softmax_xb_f32_fw, intrinsics_avx2_batch8_sum_to_one) {
//    const uint32_t x = 1000, b = 8;
//
//    auto input  = memory::allocate({memory::format::xb_f32, {b, {x}}});
//    auto output = memory::allocate({memory::format::xb_f32, {b, {x}}});
//    auto& input_memory  = input.as<const memory&>();
//    auto output_memory_ptr = output.as<const memory&>().pointer<float>();
//
//    // Initialize input data
//    fill<float>(input_memory);
//
//    auto softmax = normalization::softmax::create({output, input});
//
//    execute({output, softmax}).wait();
//
//    // Addition per batch
//    bool result = true;
//    for(uint32_t b_idx = 0; b_idx < b; ++b_idx) {
//        float sum = 0;
//        for(uint32_t x_idx = 0; x_idx < x; ++x_idx) {
//            sum += output_memory_ptr[x_idx * b + b_idx];
//        }
//        result = tests::are_equal(sum, 1.0f);
//        if(!result) break;
//    }
//
//    EXPECT_EQ(true, result);
//}
//
//TEST(softmax_xb_f32_fw, intrinsics_avx2_batch8_ref_compare) {
//    const uint32_t x = 100, b = 8;
//
//    // Optimized data
//    auto input  = memory::allocate({ memory::format::xb_f32, {b, {x}} });
//    auto output = memory::allocate({ memory::format::xb_f32, {b, {x}} });
//    auto& input_memory  = input.as<const memory&>();
//    auto& output_memory = output.as<const memory&>();
//
//    // Reference data
//    auto ref_output = memory::allocate({ memory::format::xb_f32,{b, {x}} });
//    auto& ref_output_memory = ref_output.as<const memory&>();
//
//    // Initialize input data
//    fill<float>(input_memory);
//
//    // Softmax primitives
//    auto opt_softmax = normalization::softmax::create({ output, input });
//    auto ref_softmax = normalization::softmax::create({ ref_output, input });
//
//    execute({output, opt_softmax}).wait();
//    execute({ref_output, ref_softmax}).wait();
//
//    for(uint32_t output_element = 0; output_element < output_memory.count(); ++output_element)
//        EXPECT_EQ(true, tests::are_equal(get_value<float>(ref_output_memory, output_element), get_value<float>(output_memory, output_element)));
//}
//
//TEST(softmax_xb_f32_fw, intrinsics_avx2_batch48_sum_to_one) {
//    const uint32_t x = 1000, b = 48;
//
//    auto input  = memory::allocate({memory::format::xb_f32, {b, {x}}});
//    auto output = memory::allocate({memory::format::xb_f32, {b, {x}}});
//    auto& input_memory  = input.as<const memory&>();
//    auto output_memory_ptr = output.as<const memory&>().pointer<float>();
//
//    // Initialize input data
//    fill<float>(input_memory);
//
//    auto softmax = normalization::softmax::create({output, input});
//
//    execute({output, softmax}).wait();
//
//    // Addition per batch
//    bool result = true;
//    for(uint32_t b_idx = 0; b_idx < b; ++b_idx) {
//        float sum = 0;
//        for(uint32_t x_idx = 0; x_idx < x; ++x_idx) {
//            sum += output_memory_ptr[x_idx * b + b_idx];
//        }
//        result = tests::are_equal(sum, 1.0f);
//        if(!result) break;
//    }
//
//    EXPECT_EQ(true, result);
//}
//
//TEST(softmax_xb_f32_fw, intrinsics_avx2_batch48_ref_compare) {
//    const uint32_t x = 100, b = 48;
//
//    // Optimized data
//    auto input  = memory::allocate({ memory::format::xb_f32, {b, {x}} });
//    auto output = memory::allocate({ memory::format::xb_f32, {b, {x}} });
//    auto& input_memory  = input.as<const memory&>();
//    auto& output_memory = output.as<const memory&>();
//
//    // Reference data
//    auto ref_output = memory::allocate({ memory::format::xb_f32,{b, {x}} });
//    auto& ref_output_memory = ref_output.as<const memory&>();
//
//    // Initialize input data
//    fill<float>(input_memory);
//
//    // Softmax primitives
//    auto opt_softmax = normalization::softmax::create({ output, input });
//    auto ref_softmax = normalization::softmax::create({ ref_output, input });
//
//    execute({output, opt_softmax}).wait();
//    execute({ref_output, ref_softmax}).wait();
//
//    for(uint32_t output_element = 0; output_element < output_memory.count(); ++output_element)
//        EXPECT_EQ(true, tests::are_equal(get_value<float>(ref_output_memory, output_element), get_value<float>(output_memory, output_element)));
//}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO:
//TEST(NegativeSoftmaxTest, DISABLED_TestAll) {
//}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class softmax_test : public tests::generic_test
{

public:
    softmax_test() : tests::generic_test()
    {
    }

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
        all_layer_params.push_back(new softmax("softmax", "input0"));

        //The test checks only valid combinations.
        //TODO: add more combinations.

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params()
    {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    virtual bool is_format_supported(cldnn::format format) override
    {
        return
            format == cldnn_format_type::cldnn_format_yxfb ||
            format == cldnn_format_type::cldnn_format_bfyx;
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<memory> & inputs)
    {
        assert(inputs.size() == 1);
        const memory & input = inputs[0];

        //Output is bfyx
        auto output = memory::allocate(engine, cldnn::layout(input.get_layout().data_type, cldnn::format::bfyx, input.get_layout().size));

//        const auto params = static_cast<cldnn::softmax *>(layer_parmas);

        const auto in0_mem = input.pointer<Type>();
        auto out_mem = output.pointer<Type>();

        const int in0_b = input.get_layout().size.sizes()[0];
        const int in0_f = input.get_layout().size.sizes()[1];
        const int in0_h = input.get_layout().size.sizes()[3];
        const int in0_w = input.get_layout().size.sizes()[2];

//        const int out_b = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
//        const int out_f = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
//        const int out_h = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
//        const int out_w = output.get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

//        assert(in0_b == out_b);
//        assert(in0_f == out_f);
//        assert(in0_h == out_h);
//        assert(in0_w == out_w);

        std::vector<float> cached_exp_vals;
        cached_exp_vals.resize(in0_f);

        for (int n = 0; n < in0_b; ++n)
        for (int y = 0; y < in0_h; ++y)
        for (int x = 0; x < in0_w; ++x)
        {
            float max_val = -std::numeric_limits<float>::infinity();

            for (int c = 0; c < in0_f; ++c)
            {
                const size_t in0_idx = get_linear_index(input.get_layout(), n, c, y, x);

                max_val = std::max(max_val, static_cast<float>(in0_mem[in0_idx]));
            }

            float Z = 0;

            for (int c = 0; c < in0_f; ++c)
            {
                const size_t in0_idx = get_linear_index(input.get_layout(), n, c, y, x);

                float tmp = static_cast<float>((Type)std::exp(static_cast<float>(in0_mem[in0_idx]) - max_val));
                Z += tmp;
                cached_exp_vals[c] = tmp;
            }

            for (int c = 0; c < in0_f; ++c)
            {
                const size_t out_idx = get_linear_index(output.get_layout(), n, c, y, x);
                out_mem[out_idx] = (Type)(cached_exp_vals[c] / Z);
            }
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<memory> & inputs) override
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

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info)
    {
        std::stringstream res;

        const auto & p = std::get<0>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i)
        {
            const auto chans = format::traits(p->fmt).order;

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].sizes(p->fmt).size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].sizes(p->fmt)[j];
            }
        }

        return res.str();
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;

};

std::vector<cldnn::primitive*> softmax_test::all_layer_params = {};
std::vector<tests::test_params*> softmax_test::all_generic_params = {};

TEST_P(softmax_test, DISABLED_TestAll)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(SOFTMAX,
    softmax_test,
    ::testing::Combine(::testing::ValuesIn(softmax_test::generate_generic_test_params()), ::testing::ValuesIn(softmax_test::generate_specific_test_params())),
    softmax_test::custom_param_name);

