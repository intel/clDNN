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
#include "api/CPP/convolution.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include <api/CPP/data.hpp>
#include "api/CPP/batch_norm.hpp"
#include "api/CPP/scale_grad_input.hpp"
#include "api/CPP/softmax.hpp"
#include "api/CPP/batch_norm_grad.hpp"
#include "api/CPP/mutable_data.hpp"
#include "api/CPP/convolution_grad_weights.hpp"
#include "api/CPP/convolution_grad_input.hpp"
#include "api/CPP/scale_grad_weights.hpp"
#include "api/CPP/scale.hpp"
#include "api_extension/CPP/fused_conv_bn_scale.hpp"
#include "test_utils/test_utils.h"

#include <algorithm>
#include <thread>
#include <fstream>


using namespace cldnn;
using namespace tests;

TEST(fused_conv_bn_scale_gpu, base) {

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
    auto scale_in = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto scale_bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values<float>(input, { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f });
    set_values<float>(weights, { 8.1f, 7.2f, 6.3f, 5.4f, 4.5f, 3.6f, 2.7f, 1.8f });
    set_values<float>(bias, { 5.6f, 2.5f });
    set_values<float>(scale_in, { 1.3f, 1.4f });
    set_values<float>(scale_bias, { 3.5f, 5.3f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("bias", bias),
        data("scale_in", scale_in),
        data("scale_bias", scale_bias),
        fused_conv_bn_scale("fuse", "input", { "weights" }, { "bias" }, 0.000001f, "scale_in", "scale_bias")
    );

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));

    auto output_prim = outputs.at("fuse").get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected = {
        2.2f, 3.9f, 4.8f, 6.7f
    };

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(output_ptr[i], expected[i]));
    }
}

TEST(fused_conv_bn_scale_gpu, fusing_scale_batch_norm) {

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
    auto scale_in = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto scale_bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values<float>(input, { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f });
    set_values<float>(weights, { 8.1f, 7.2f, 6.3f, 5.4f, 4.5f, 3.6f, 2.7f, 1.8f });
    set_values<float>(bias, { 5.6f, 2.5f });
    set_values<float>(scale_in, { 1.3f, 1.4f });
    set_values<float>(scale_bias, { 3.5f, 5.3f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("bias", bias),
        data("scale_in", scale_in),
        data("scale_bias", scale_bias),
        convolution("conv", "input", { "weights" }, { "bias" }),
        batch_norm("bn", "conv", 0.000001f),
        scale("scale", "bn", "scale_in", "scale_bias")
    );

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto executed = network.get_executed_primitive_ids();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_TRUE(std::find(executed.begin(), executed.end(), "conv") == executed.end());
    EXPECT_TRUE(std::find(executed.begin(), executed.end(), "bn") == executed.end());

    auto output_prim = outputs.at("scale").get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected = {
        2.2f, 3.9f, 4.8f, 6.7f
    };

    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(are_equal(output_ptr[i], expected[i]));
    }
}

TEST(fused_conv_bn_scale_gpu, fusing_scale_batch_norm_forward_backward) {

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
    auto scale_in = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto scale_bias = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto weights_2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1} });
    auto inv_var = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values<float>(input, { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f });
    set_values<float>(weights, { 8.1f, 7.2f, 6.3f, 5.4f, 4.5f, 3.6f, 2.7f, 1.8f });
    set_values<float>(bias, { 5.6f, 2.5f });
    set_values<float>(scale_in, { 1.3f, 1.4f });
    set_values<float>(scale_bias, { 3.5f, 5.3f });
    set_values(weights_2, { 1.f, 1.f });

    topology topology(
        input_layout("input", input.get_layout()),
        mutable_data("weights", weights),
        mutable_data("bias", bias),
        mutable_data("scale_in", scale_in),
        mutable_data("scale_bias", scale_bias),
        mutable_data("inv_var", inv_var),
        mutable_data("weights_2", weights_2),
        convolution("conv", "input", { "weights" }, { "bias" }),
        batch_norm("bn", "conv", 0.000001f, "inv_var"),
        scale("scale", "bn", "scale_in", "scale_bias"),
        convolution("conv_2", "scale", { "weights_2" }),
        softmax("softmax", "conv_2"),
        convolution_grad_input("conv_grad_in_2", "softmax", { "weights_2" }),
        scale_grad_input("scale_grad_in", "conv_grad_in_2", "scale_in"),
        scale_grad_weights("scale_grad_w", "bn", "conv_grad_in_2", "scale_in", "scale_bias", "scale_grad_in"),
        batch_norm_grad("bn_grad", "scale_grad_in", "conv", "inv_var"),
        convolution_grad_input("conv_grad_in", "bn_grad", { "weights" }),
        convolution_grad_weights("conv_grad_w", "bn_grad", "input", { "weights" }, { "bias" }, {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, "conv_grad_in")
    );

    build_options bo;
    bo.set_option(build_option::outputs({ "conv_grad_in", "conv_grad_w", "scale_grad_w" }));
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto executed = network.get_executed_primitive_ids();
    EXPECT_EQ(outputs.size(), size_t(3));
    EXPECT_TRUE(std::find(executed.begin(), executed.end(), "bn") == executed.end());
    EXPECT_TRUE(std::find(executed.begin(), executed.end(), "scale") == executed.end());

    auto output_prim = outputs.at("conv_grad_in").get_memory();

    auto output_ptr = output_prim.pointer<float>();
    
    std::vector<float> expected = {
        -63.f, -54.f, -45.f, -36.f, -63.f, -54.f, -45.f, -36.f
    };

    for (int i = 0; i < static_cast<int>(output_ptr.size()); i++)
    {
        EXPECT_TRUE(are_equal(output_ptr[i], expected[i]));
    }
}