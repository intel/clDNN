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

#include "layout_optimizer.h"
#include "topology_impl.h"
#include "network_impl.h"
#include "primitive_inst.h"

#include "api/CPP/data.hpp"
#include "api/CPP/reorder.hpp"
#include <boost/filesystem.hpp>
#include <sstream>

using namespace cldnn;

layout_optimizer::layout_optimizer(refcounted_obj_ptr<engine_impl> eng, bool enabled)
    : _enabled(enabled), _topology(), _engine(eng), _optimization_attributes()
{
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const convolution>, boost::optional<layout> const& output_layout)
{
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;
    auto batch = current_layout.size.batch[0];

    if (output_layout)
    {
        expected_data_type = output_layout.get().data_type;
        batch = output_layout.get().size.batch[0];
    }
    else if (type != data_type::input)
        throw std::runtime_error("'output_layout' is required parameter for weights/bias optimization");

    switch (type)
    {
    case data_type::bias: //convolution bias
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    case data_type::weights: //convolution weights
        if (batch < 32 || expected_data_type != data_types::f16 || !_optimization_attributes.splitted_convolution)
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::os_iyx_osv16;
        }
        else
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::yxfb;
        }

        break;

    case data_type::input: //convolution input
        if (current_layout.format.dimension() != 4)
            throw std::runtime_error("Convolution input not 4-dimensional?");

        if (expected_data_type != data_types::f16 || batch < 32 || !_optimization_attributes.splitted_convolution)
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::bfyx;
        }
        else
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::yxfb;
        }

        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for convolution primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const fully_connected>, boost::optional<layout> const& output_layout)
{
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;
    auto batch = current_layout.size.batch[0];

    if (output_layout)
    {
        expected_data_type = output_layout.get().data_type;
        batch = output_layout.get().size.batch[0];
    }
    else if (type != data_type::input)
        throw std::runtime_error("'output_layout' is required parameter for weights/bias optimization");

    switch (type)
    {
    case data_type::bias: //fc bias
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    case data_type::weights: //fc weights
    {
        if (batch > 1 && expected_data_type != data_types::f16 && batch % 8 == 0)
        {
            expected_tensor = cldnn::tensor(
                current_layout.size.batch[0], 1, current_layout.size.feature[0] * current_layout.size.spatial[0] * current_layout.size.spatial[1], 1
            );
            expected_format = cldnn::format::bs_xs_xsv8_bsv8;
        }
        else if (batch == 1)
        {
            expected_tensor = cldnn::tensor(
                current_layout.size.batch[0], 1, current_layout.size.feature[0] * current_layout.size.spatial[0] * current_layout.size.spatial[1], 1
            );
            expected_format = cldnn::format::bs_x_bsv16;
        }
        else
        {
            expected_tensor = current_layout.size;
            expected_format = cldnn::format::yxfb;
        }

        break;
    }

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for fully-connected primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout, data_type type, std::shared_ptr<const deconvolution>, boost::optional<layout> const& output_layout)
{
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (output_layout)
    {
        expected_data_type = output_layout.get().data_type;
    }
    else if (type != data_type::input)
        throw std::runtime_error("'output_layout' is required parameter for weights/bias optimization");

    switch (type)
    {
    case data_type::bias: //convolution bias
        expected_tensor = cldnn::tensor(1, 1, static_cast<tensor::value_type>(current_layout.count()), 1);
        expected_format = cldnn::format::bfyx;
        break;

    case data_type::weights: //convolution weights
        expected_tensor = current_layout.size;
        expected_format = cldnn::format::bfyx;
        break;

    default:
        throw std::runtime_error("Unsupported data type in layout_optimizer::get_expected_layout for convolution primitive");
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

std::pair<std::shared_ptr<cldnn::reorder>, bool>
layout_optimizer::create_reorder_if_needed(const layout& current_layout, const cldnn::primitive_id& memid, layout const& expected_layout)
{
    if (current_layout != expected_layout)
    {
        cache_key ckey{ memid, expected_layout };
        auto itr = _cached_reorders.find(ckey);
        if (itr != _cached_reorders.end())
            return std::make_pair(itr->second, true);

        auto count = _cached_reorders.size();
        std::stringstream ss;
        ss << "reorder_" << count << "_" << memid;

        auto reorder = std::make_shared<cldnn::reorder>(ss.str(), memid, expected_layout);
        _cached_reorders[ckey] = reorder;
        return std::make_pair(reorder, false);
    }

    return std::make_pair(nullptr, true);
}

void layout_optimizer::optimize() const
{
    if (!_enabled)
        return;

    network_impl net(_engine, _topology);
    net.execute(std::vector<refcounted_obj_ptr<event_impl>>());
    for (auto const& output : net.get_outputs())
    {
        auto input_id = output->dependencies().at(0)->id();
        auto& data_node = net.get_program()->get_node(input_id).as<data>();
        const_cast<data&>(*data_node.get_primitive()).mem = output->output_memory();
    }
}
