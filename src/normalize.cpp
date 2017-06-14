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

#include "normalize_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id normalize_type_id()
{
    static primitive_type_base<normalize> instance;
    return &instance;
}

layout normalize_inst::calc_output_layout(normalize_node const& node)
{
    return node.input().get_output_layout();
}

std::string normalize_inst::to_string(normalize_node const& node)
{
    std::stringstream           primitive_description;
    auto desc = node.get_primitive();
    auto input = node.input();
    auto scale_input = node.scale();
    auto epsilon = desc->epsilon;
    auto norm_region = desc->across_spatial ? "across spatial" : "within spatial";

    primitive_description << "id: " << desc->id << ", type: normalize" <<
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tscale input: " << scale_input.id() << ", count: " << scale_input.get_output_layout().count() << ",  size: " << scale_input.get_output_layout().size <<
        "\n\tepsilon: " << epsilon << ", normalization region: " << norm_region <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

normalize_inst::typed_primitive_inst(network_impl& network, normalize_node const& node)
    :parent(network, node)
{
    /// Scale x dimension should be 1 (if all channels have the same scale) or equal to input feature size (one scale per channel).
    auto scale_size = scale_memory().get_layout().size;
    auto scale_feature_size = scale_size.spatial[0];
    auto input_feature_size = input_memory().get_layout().size.feature[0];

    if ((scale_feature_size != 1) && (scale_feature_size != input_feature_size))
    {
        throw std::invalid_argument("Dimensions mismatch between input and scale input in Normalize layer!");
    }

    // All other dimensions should be 1
    if((int32_t)scale_size.count() != scale_feature_size)
    {
        throw std::invalid_argument("Dimensions mismatch of scale input in Normalize layer!");
    }

}
}
