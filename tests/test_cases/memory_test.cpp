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
#include <api/CPP/engine.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/activation.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

#if 0
TEST(memory_tests, DISABLED_execution_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in.get_layout()),
        activation("out", "in", activation_linear)
    };

    network net(eng, tpl);
    
    while (true)
    {
        net.set_input_data("in", in);
        net.execute();
    }
}

TEST(memory_tests, DISABLED_network_creation_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx,{ 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in.get_layout()),
        activation("out", "in", activation_linear)
    };

    while (true)
    {
        network net(eng, tpl);
    }
}
#endif
