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
#include "deconvolution_inst.h"
#include "primitive_type_base.h"

namespace cldnn
{
primitive_type_id deconvolution_type_id()
{
    static primitive_type_base<deconvolution, deconvolution_inst> instance;
    return &instance;
}

layout deconvolution_inst::calc_output_layout(deconvolution_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout(); //weights are stored after inputs
    auto input_offset = desc->input_offset;
    auto strd = desc->stride;
    auto split = desc->weights.size();

    //compute output_dim <= stride * (input_size - 1) + kernel_size + 2 * input_offset;
    auto kernel_xy = weights_layout.size.spatial;
    if (kernel_xy.size() != 2) 
        throw std::runtime_error("Weights have to have 2 dimensions in spatial domain.");

    auto output_spatial_x = strd.spatial[0] * (input_layout.size.spatial[0] - 1) + kernel_xy[0] + 2 * input_offset.spatial[0];
    auto output_spatial_y = strd.spatial[1] * (input_layout.size.spatial[1] - 1) + kernel_xy[1] + 2 * input_offset.spatial[1];
    auto number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(split);

    tensor output_size(input_layout.size.batch[0], number_of_features, output_spatial_x, output_spatial_y);

    auto result = layout({ input_layout.data_type, input_layout.format, output_size });
    return result;
}

std::string deconvolution_inst::to_string(deconvolution_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto input                  = node.input();
    auto strd                   = desc->stride;
    auto activation             = desc->with_activation ? " true" : "false";
    std::stringstream           ss_weights, ss_biases;
    for (size_t i = 0; i < desc->weights.size(); ++i)
    {
        ss_weights << node.weights(i).id();
        ss_weights << ", count: " << node.weights(i).get_output_layout().count();
        i != (desc->weights.size() - 1) ? ss_weights << ", " : ss_weights << "";
    }

    for (size_t i = 0; i < desc->bias.size(); ++i)
    {
        ss_biases << node.bias(i).id();
        ss_biases << ", count: " << node.bias(i).get_output_layout().count();
        i != (desc->bias.size() - 1) ? ss_biases << ", " : ss_biases << "";
    }

    primitive_description << "id: " << desc->id << ", type: deconvolution" << 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ",  size: " << input.get_output_layout().size <<
        "\n\tweights count: " << desc->weights.size() << 
        "\n\tweights: " << ss_weights.str() << 
        "\n\tbiases count: " << desc->bias.size() <<
        "\n\tbiases: " << ss_biases.str() << 
        "\n\tstride: " << strd.spatial[0] << "x" << strd.spatial[1] <<
        "\n\twith activation: " << activation << ", slope: " << desc->activation_negative_slope <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

deconvolution_inst::typed_primitive_inst(network_impl& network, deconvolution_node const& node)
    : parent(network, node)
{
    auto stride = argument.stride;
    auto output_size = output_memory().get_layout().size;

    auto input_inst = input_memory().get_layout();
    auto output_inst = output_memory().get_layout();

    if (input_inst.size.raw.size() != output_inst.size.raw.size())
        throw std::runtime_error("Input/output number of dimension does not match.");
    if (stride.raw.size() != output_inst.size.raw.size())
        throw std::runtime_error("Stride/output number of dimension does not match.");

    auto split = argument.split();
    for (decltype(split) j = 0; j < split; j++)
    {
        auto& filter_mem = weights_memory(j);
        auto& filter_inst = filter_mem.get_layout(); //deconvolution filter
        auto& bias_inst = bias_memory(j).get_layout();

        auto input_offset = argument.input_offset;

        if (bias_inst.size.batch[0] != 1 && bias_inst.size.feature[0] != 1 && bias_inst.size.spatial[1] != 1)
            throw std::runtime_error("Biases isn't 1D vector."); // b=1, f=1
        if (bias_inst.size.spatial[0] != output_size.feature[0] / split)
            throw std::runtime_error("Biases/output feature maps number does not match.");
        if (node.get_output_layout().data_padding.filling_value() != 0.0f)
            throw std::runtime_error("Wnknown padding mode.");
        if (input_offset.raw.size() != input_inst.size.raw.size())
            throw std::runtime_error("Input offset/input number of dimension does not match.");
        if (1 != output_size.feature.size())
            throw std::runtime_error("Only one-dimensional features are supported");
        if (1 != output_size.batch.size())
            throw std::runtime_error("Only one-dimensional batch size is supported");
        if (2 != filter_inst.size.spatial.size())
            throw std::runtime_error("Weights have to have 2 dimensions in spatial domain.");

        if ((input_inst.size.feature[0] - input_offset.feature[0]) / split < filter_inst.size.feature[0])
            throw std::runtime_error("Weights/input feature maps number does not match.");
    }
}
}
