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
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>

#include <api/CPP/data.hpp>
#include <api/CPP/reshape.hpp>
#include <api/CPP/input_layout.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;
using namespace testing;

template <class ElemType>
void generic_reshape_test(format fmt, tensor const& input_size, tensor const& reshape_size, bool in_place, padding const& input_padd = padding(), padding const& output_padd = padding())
{
    engine engine;

    //allocate input memory
    auto input = memory::allocate(engine, { std::is_same<ElemType, FLOAT16>::value ? data_types::f16 : data_types::f32, fmt, input_size });
    
    {
        auto input_ptr = input.cldnn::memory::pointer<ElemType>();
        auto input_itr = input_ptr.begin();

        auto elements = input_size.count();

        int value = 1;
        for (size_t i = 0; i < elements; ++i)
            *input_itr++ = (ElemType)value++;
    }

    topology tpl;
    std::string reshape_input = "input";

    tpl.add(input_layout("input", input.get_layout()));
    if (input_padd)
    {
        auto padded_input_layout = input.get_layout();
        padded_input_layout.data_padding = input_padd;
        tpl.add(reorder("reorder", "input", padded_input_layout));
        reshape_input = "reorder";
    }
    tpl.add(reshape("reshape", reshape_input, reshape_size, output_padd));

    build_options bo;
    bo.set_option(build_option::outputs({ reshape_input, "reshape" }));

    network net(engine, tpl, bo);
    net.set_input_data("input", input);
    auto outputs = net.execute();

    ASSERT_TRUE(outputs.size() == 2 && outputs.count("reshape") == 1 && outputs.count(reshape_input) == 1);
    auto net_input = outputs.at(reshape_input).get_memory();
    auto output = outputs.at("reshape").get_memory();

    EXPECT_TRUE(output.get_layout().data_type == input.get_layout().data_type);     //reshape should not change data_type
    EXPECT_TRUE(output.get_layout().format == input.get_layout().format); //reshape should not change format

    //output size should be equal to requested plus output padding
    ASSERT_TRUE(output.get_layout().size == reshape_size);
    ASSERT_TRUE(output.get_layout().get_buffer_size() == reshape_size.add(output_padd.lower_size()).add(output_padd.upper_size()));

    if (in_place)
        EXPECT_TRUE(output.is_the_same_buffer(net_input)); //if reshape should operate in place both memories should refer to the same underlaying cl::Buffer
    else
        EXPECT_TRUE(!output.is_the_same_buffer(net_input)); //otherwise they should not

    {
        auto output_ptr = output.pointer<const ElemType>();
        auto output_itr = output_ptr.begin();

        auto sizes = reshape_size.sizes(fmt);
        auto lower = output_padd.lower_size().sizes(fmt);
        auto upper = output_padd.upper_size().sizes(fmt);
        auto buffer_sizes = sizes;
        int32_t accum = 1;
        for (size_t i = 1; i <= sizes.size(); ++i)
        {
            buffer_sizes[sizes.size() - i] = accum;
            accum *= lower[sizes.size() - i] + sizes[sizes.size() - i] + upper[sizes.size() - i];
        }

        int value = 1;

        output_itr += lower[0] * buffer_sizes[0];
        for (int d1 = 0; d1 < sizes[0]; ++d1)
        {
            output_itr += lower[1] * buffer_sizes[1];
            for (int d2 = 0; d2 < sizes[1]; ++d2)
            {
                output_itr += lower[2] * buffer_sizes[2];
                for (int d3 = 0; d3 < sizes[2]; ++d3)
                {
                    output_itr += lower[3] * buffer_sizes[3];
                    for (int d4 = 0; d4 < sizes[3]; ++d4)
                    {
                        auto& output_value = *output_itr;
                        ++output_itr;
                        EXPECT_FLOAT_EQ(output_value, (ElemType)value);
                        ++value;
                    }

                    output_itr += upper[3] * buffer_sizes[3];
                }

                output_itr += upper[2] * buffer_sizes[2];
            }

            output_itr += upper[1] * buffer_sizes[1];
        }
    }
}

TEST(reshape_gpu_f32, basic_2dim_in_place)
{
    generic_reshape_test<float>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 4, 1),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_in_place)
{
    generic_reshape_test<FLOAT16>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_in_place)
{
    generic_reshape_test<float>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(27, 2, 3, 4),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_in_place)
{
    generic_reshape_test<FLOAT16>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshpape_gpu_f32, basic_2dim_output_padd)
{
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 4, 2),
        tensor(1, 1, 8, 1),
        false,
        padding(),
        padding(std::vector<int>{ 0,0,1,1 })
        );
}

TEST(reshape_gpu_f16, basic_2dim_output_padd)
{
    generic_reshape_test<FLOAT16>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{ 0,0,2,2 })
        );
}

TEST(reshape_gpu_f32, basic_2dim_input_padd)
{
    generic_reshape_test<float>(
        format::fyxb,
        tensor(1, 1, 2, 5),
        tensor(1, 1, 5, 2),
        false,
        padding({ 0,0,3,2 }, { 0,0,1,4 })
        );
}

TEST(reshape_gpu_f16, basic_2dim_input_padd)
{
    generic_reshape_test<FLOAT16>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({ 0,0,4,1 }, { 0,0,2,3 })
        );
}

TEST(reshape_gpu_f32, basic_2dim_input_output_padd)
{
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({ 0,0,4,4 }, { 0,0,1,1 }),
        padding({ 0,0,0,0 }, { 0,0,3,0 })
        );
}

TEST(reshape_gpu_f16, basic_2dim_input_output_padd)
{
    generic_reshape_test<FLOAT16>(
        format::byxf,
        tensor(1, 1, 6, 6),
        tensor(1, 1, 3, 12),
        false,
        padding({ 0,0,1,1 }, { 0,0,0,0 }),
        padding({ 0,0,2,1 }, { 0,0,1,2 })
        );
}

TEST(reshpape_gpu_f32, basic_4dim_output_padd)
{
    generic_reshape_test<float>(
        format::bfyx,
        tensor(2, 5, 7, 3),
        tensor(1, 14, 15, 1),
        false,
        padding(),
        padding({ 1,0,0,1 },{ 0,2,3,0 })
        );
}

TEST(reshape_gpu_f16, basic_4dim_output_padd)
{
    generic_reshape_test<FLOAT16>(
        format::bfyx,
        tensor(5, 4, 2, 2),
        tensor(40, 2, 1, 1),
        false,
        padding(),
        padding({ 0,2,0,1 },{ 0,2,3,0 })
        );
}

TEST(reshape_gpu_f32, basic_4dim_input_padd)
{
    generic_reshape_test<float>(
        format::yxfb,
        tensor(8, 128, 3, 3),
        tensor(16, 8, 8, 9),
        false,
        padding({ 0,1,3,3}, { 0,1,1,1 })
        );
}

TEST(reshape_gpu_f16, basic_4dim_input_padd)
{
    generic_reshape_test<FLOAT16>(
        format::yxfb,
        tensor(2, 32, 8, 8),
        tensor(8, 128, 1, 4),
        false,
        padding({ 2,2,1,0 }, { 1,2,2,0 })
        );
}

TEST(reshape_gpu_f32, basic_4dim_input_output_padd)
{
    generic_reshape_test<float>(
        format::fyxb,
        tensor(8, 1024, 25, 25),
        tensor(8, 64, 100, 100),
        false,
        padding({ 2,0,2,1 }, { 0,1,4,0 }),
        padding({ 1,2,3,4 }, { 0,4,1,1 })
        );
}

TEST(reshape_gpu_f16, basic_4dim_input_output_padd)
{
    generic_reshape_test<FLOAT16>(
        format::byxf,
        tensor(32, 3, 227, 227),
        tensor(8, 12, 227, 227),
        false,
        padding({ 0,1,4,4 }, { 0,1,1,1 }),
        padding({ 0,29,29,0 }, { 0,0,0,0 })
        );
}