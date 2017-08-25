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

#include "batch_norm_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"

namespace cldnn
{
primitive_type_id batch_norm_type_id()
{
    static primitive_type_base<batch_norm> instance;
    return &instance;
}

layout batch_norm_inst::calc_output_layout(batch_norm_node const& node)
{
    return node.input().get_output_layout();
}

std::string batch_norm_inst::to_string(batch_norm_node const& node)
{
    std::stringstream               primitive_description;
    auto desc                       = node.get_primitive();
    auto& input                     = node.input();
    auto& mean                      = node.mean();
    auto& variance                  = node.variance();
    auto global_stats               = desc->use_global_stats ? " true" : "false";

    primitive_description << "id: " << desc->id << ", type: batch_norm" << 
        "\n\tinput id: " << input.id() << ", size: " << node.input().get_output_layout().count() << ",  size: " << input.get_output_layout().size <<
        "\n\tmean id: " << mean.id() << ", size: " << mean.get_output_layout().count() << ",  size: " << mean.get_output_layout().size <<
        "\n\tvariance id: " << variance.id() << ", size: " << variance.get_output_layout().count() << ",  size: " << variance.get_output_layout().size <<
        "\n\tuse_global_stats: " << global_stats << 
        "\n\tepsilon: " << desc->epsilon << 
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

batch_norm_inst::typed_primitive_inst(network_impl& network, batch_norm_node const& node)
    :parent(network, node) 
{
    auto mean_format = mean_memory().get_layout().format;
    auto variance_format = variance_memory().get_layout().format;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Mean format", mean_format.value, "supported mean formats", format::yxfb, format::bfyx );
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Variance format", variance_format.value, "supported variance formats", format::yxfb, format::bfyx );
}
}