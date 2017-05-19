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
#include "api/CPP/fully_connected.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/data.hpp>

namespace cldnn
{
	template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;

cldnn::format::type layout_4d(cldnn::format f) {
	switch (f.value) {
	case cldnn::format::bfyx:
		return cldnn::format::bfyx;
	case cldnn::format::yxfb:
		return cldnn::format::yxfb;
	default:
		return f.value;
	}
}

template <typename T>
VVVVF<T> fully_connected_reference(VVVVF<T> &input, VVVVF<T> &weights, VF<T> &bias, bool relu = false, T slope = 0.0f) {
	size_t input_f = input[0].size();
	size_t input_y = input[0][0].size();
	size_t input_x = input[0][0][0].size();
	size_t output_b = input.size();		// input is assumed to be bfyx
	size_t output_x = weights.size();	// weights is assumed to be bfyx
	VVVVF<T> output(output_b, VVVF<T>(1, VVF<T>(1, VF<T>(output_x))));
	T res;
	for (size_t b = 0; b < output_b; ++b) {
		for (size_t n = 0; n < output_x; ++n) {
			res = bias[n];
			for (size_t f = 0; f < input_f; ++f) {
				for (size_t y = 0; y < input_y; ++y) {
					for (size_t x = 0; x < input_x; ++x) {
						res += input[b][f][y][x] * weights[n][f][y][x];
					}
				}
			}
			if (relu && res < (T)0)
				res *= slope;
			output[b][0][0][n] = res;
		}
	}
	return output;
}

template <typename T>
void generic_fully_connected_test(cldnn::format test_input_fmt, cldnn::format test_weights_fmt, int input_b, int f, int y, int x, int output_x, bool relu, T slope) {
	int min_random = -2, max_random = 2;
	VVVVF<T> input_rnd = generate_random_4d<T>(input_b, f, y, x, min_random, max_random);
	VVVVF<T> weights_rnd = generate_random_4d<T>(output_x, f, y, x, min_random, max_random);
	VF<T> bias_rnd_vec = generate_random_1d<T>(output_x, min_random, max_random);
	VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
	VF<T> weights_rnd_vec = flatten_4d<T>(test_weights_fmt, weights_rnd);

	engine engine;
	tensor input_tensor( input_b, f, x, y );
	tensor weights_tensor( output_x, f, x, y );
	auto input = memory::allocate(engine, { type_to_data_type<T>::value, test_input_fmt, input_tensor });
	auto weights = memory::allocate(engine, { type_to_data_type<T>::value, test_weights_fmt, weights_tensor });
	auto bias = memory::allocate(engine, { type_to_data_type<T>::value, format::bfyx, { 1,1,output_x,1 } });
	set_values(input, input_rnd_vec);
	set_values(weights, weights_rnd_vec);
	set_values(bias, bias_rnd_vec);

	topology topology(
		input_layout("input", input.get_layout()),
		data("weights", weights),
		data("bias", bias),
		fully_connected("fully_connected", "input", "weights", "bias", relu, slope)
	);

	network network(engine, topology);
	network.set_input_data("input", input);

	auto outputs = network.execute();
	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "fully_connected");

	auto output_memory = outputs.at("fully_connected").get_memory();
	auto output_layout = output_memory.get_layout();
	auto output_ptr = output_memory.pointer<T>();

	//EXPECT_EQ(output_layout.format.value, test_input_fmt);
	tensor output_tensor = output_layout.size;
	int b_size = output_tensor.batch[0];
	int x_size = output_tensor.spatial[0];
	EXPECT_EQ(b_size, input_b);
	EXPECT_EQ(x_size, output_x);
	
	bool test_is_correct = true;
	VVVVF<T> output_cpu = fully_connected_reference<T>(input_rnd, weights_rnd, bias_rnd_vec, relu, slope);
	VF<T> output_cpu_vec = flatten_4d<T>(layout_4d(output_layout.format), output_cpu);
	for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
		if (!floating_point_equal(output_cpu_vec[i], output_ptr[i])) {
			EXPECT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]); // to print the problematic values
			test_is_correct = false;
			break;
		}
	}
	EXPECT_EQ(test_is_correct, true) << std::endl
		<< "failing test parameters:" << std::endl
		<< "test_input_fmt = " << test_input_fmt.value << std::endl
		<< "test_weights_fmt = " << test_weights_fmt.value << std::endl
		<< "input_b = " << input_b << std::endl
		<< "f = " << f << std::endl
		<< "y = " << y << std::endl
		<< "x = " << x << std::endl
		<< "output_x = " << output_x << std::endl
		<< "relu = " << relu << std::endl
		<< "slope = " << (float)slope << std::endl
		<< "type = " << (sizeof(T) == 2 ? "float16" : "float32") << std::endl;
}

TEST(DISABLED_fully_connected_gpu, generic_random_short) {
	VF<cldnn::format> test_input_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
	VF<cldnn::format> test_weights_fmts = { cldnn::format::yxfb };
	VF<bool> relu = { true, false };
	VF<float> slopes = { 0.0f, 3.125f };
	std::vector<std::pair<int, int>> input_sizes = { { 100, 100 },{ 400, 600 },{ 531, 777 },{ 4096, 1980 } };
	VF<int> outputs_x = { 5, 16 };

	engine engine;
	bool f16_supported = !!engine.get_info().supports_fp16;
	if (!f16_supported) {
		std::cout << "[ SKIPPED  ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
	}

	for (cldnn::format test_input_fmt : test_input_fmts) {
		for (cldnn::format test_weights_fmt : test_weights_fmts) {
			for (int input_b = 1; input_b <= 16; input_b *= 2) {
				for (int input_f = 2; input_f <= 2; ++input_f) {
					for (std::pair<int, int> &input_yx : input_sizes) {
						for (int output_x : outputs_x) {
							for (bool relu_activated : relu) {
								for (float slope : slopes) {
									generic_fully_connected_test<float>(test_input_fmt, test_weights_fmt, input_b, input_f, input_yx.first, input_yx.second, output_x, relu_activated, slope);
									if (!f16_supported) continue;
									generic_fully_connected_test<FLOAT16>(test_input_fmt, test_weights_fmt, input_b, input_f, input_yx.first, input_yx.second, output_x, relu_activated, (FLOAT16)slope);
								}
							}
						}
					}
				}
			}
		}
	}
	
}

TEST(fully_connected_gpu, no_biases) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   no biases
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b, 1, input_x, 1} });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    auto input = input_layout("input", input_prim.get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("full_con_prim", "input", "weights");
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(1.5f, output_ptr[0]);
    EXPECT_EQ(0.75f, output_ptr[1]);
    EXPECT_EQ(-2.25f, output_ptr[2]);
    EXPECT_EQ(3.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_1) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_x = 4,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate( engine, { data_types::f32, format::yxfb, { input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx, { 1,1,output_x, 1} });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.5f, output_ptr[0]);
    EXPECT_EQ(2.75f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(7.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75     0.75   7
    //   4      1        2.75   5

    const int32_t output_x = 4,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b,1,input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_x,1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(2.75f, output_ptr[2]);
    EXPECT_EQ(1.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(7.00f, output_ptr[6]);
    EXPECT_EQ(5.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_x = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx, { 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_x,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(2.75f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(7.00f, output_ptr[3]);
}


TEST(fully_connected_gpu, yxfn_f32) {
    //  Input  : 1x2x1x2 - 1 batch 2 feature maps of size 2x1
    //  Output : 2x1 - 2 batches 1 neuron each
    //  Weights: 2x2x1x2 - 2 neurons with weights of 2 feature maps of size 2x1
    //
    //  Input:
    //   1  -2      f0: b0
    //   3  -4      f1: b0

    //  Weights:
    //   1  -1      n0: fm0  
    //   2   0      n0: fm1
    //   3   4      n1: fm0
    //   0.5 5      n1: fm1
    //
    //  Biases:
    //   1.0 -5
    //
    //  Output:
    //   10  -28.5

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 2, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ 2 ,{ { 1 } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 1 } });

    set_values(input_prim, { 1.f, 3.f, -2.f, -4.f });
    set_values(weights_prim, { 1.f, -1.f, 2.0f, 0.f, 3.0f, 4.0f, 0.5f, 5.0f });
    set_values(bias_prim, { 1.0f, -5.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias")
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(10, output_ptr[0]);
    EXPECT_EQ(-28.5, output_ptr[1]);
}

TEST(fully_connected_gpu, xb_f32_batch_1_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0,  -2.0,  3.0,  -4.0
    //
    //  Output:
    //   2.5   0      0.75  0

    const int32_t output_x = 4,  // size of whole output buffer
        input_x = 3, input_b = 1,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_x, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2_relu) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //
    //  Output:
    //   2.5    0   0.75   0
    //   4      0   2.75   0

    const int32_t output_x = 4,  // size of whole output buffer
        input_x = 3, input_b = 2,  // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::yxfb,{ input_b, 1, input_x, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32,{ output_b,{ { output_x } },{ 1 } } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_x,1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(4.00f, output_ptr[1]);
    EXPECT_EQ(0.00f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
    EXPECT_EQ(0.75f, output_ptr[4]);
    EXPECT_EQ(2.75f, output_ptr[5]);
    EXPECT_EQ(0.00f, output_ptr[6]);
    EXPECT_EQ(0.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   0    0.75  0

    const int32_t output_x = 4,                 // size of whole output buffer
        input_x = 3,                 // size of whole input buffer
        weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_x } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_x,1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_EQ(2.50f, output_ptr[0]);
    EXPECT_EQ(0.00f, output_ptr[1]);
    EXPECT_EQ(0.75f, output_ptr[2]);
    EXPECT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, x_f32_relu_with_negative_slope) {
	//  Input  : 3x1
	//  Output : 4x1
	//  Weights: 4x3
	//  Negative Slope: 0.1
	//
	//  Input:
	//  -0.5     2    0.5
	//
	//  Weights:
	//   1.5     1    0.5
	//  -1       0    0.5
	//   0.5    -0.5 -2
	//  -0.5     1    1.5
	//
	//  Biases:
	//   1.0, -2.0, 3.0, -4.0
	//  Output:
	//   2.5   -0.125    0.75  -0.1

	const int32_t output_x = 4,                 // size of whole output buffer
		input_x = 3,                 // size of whole input buffer
		weight_b = 4, weight_x = 3;  // size of whole weights buffer

    engine engine;

    auto input_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,input_x,1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32,{ 1       ,{ { output_x } }, 1 } });
    auto weights_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ weight_b, 1, weight_x, 1 } });
    auto bias_prim = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1,1,output_x,1 } });

	set_values(input_prim, { -0.5f, 2.0f, 0.5f });
	set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
	set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim.get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("full_con_prim", "input", "weights", "bias", true, 0.1f)
    );

    network network(engine, topology);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "full_con_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

	EXPECT_EQ(2.50f, output_ptr[0]);
	EXPECT_EQ(-0.125f, output_ptr[1]);
	EXPECT_EQ(0.75f, output_ptr[2]);
	EXPECT_EQ(-0.1f, output_ptr[3]);
}