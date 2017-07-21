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
#include "convolution_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"

namespace cldnn
{
primitive_type_id convolution_type_id()
{
    static primitive_type_base<convolution> instance;
    return &instance;
}

layout convolution_inst::calc_output_layout(convolution_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout(); //weights are stored after inputs

    auto input_offset = desc->input_offset;
    auto stride = desc->stride;
    auto dilation = desc->dilation;
    auto split = desc->weights.size();

    // compute how many outputs in rows and columns will be generate by filter. 
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride 
    auto filter_size = weights_layout.size;

    // TODO: Consider moving general parameter verification to arguments constructor.
    if (stride.spatial[0] <= 0 || stride.spatial[1] <= 0)
        throw std::invalid_argument("Stride must be positive (>= 1)");
    if (dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0)
        throw std::invalid_argument("Dilation must be positive (>= 1)");
    if (2 * input_offset.spatial[0] > input_layout.size.spatial[0] || 2 * input_offset.spatial[1] > input_layout.size.spatial[1])
        throw std::invalid_argument("Input offset is greater than input data range. There is no input data to process");

    // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
    auto number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(split);

    if (desc->with_output_size)
    {
        if (desc->output_size.spatial[0] <= 0 || desc->output_size.spatial[1] <= 0)
            throw std::invalid_argument("User-defined size of output layout must be positive (>= 1)");

        tensor output_size(input_layout.size.batch[0], number_of_features,
                           desc->output_size.spatial[0], desc->output_size.spatial[1]);
        return { input_layout.data_type, input_layout.format, output_size };
    }

    auto output_range = calc_sliding_window_output_range<swor_mode::all>(
        input_layout.size, filter_size, input_offset, stride, dilation, true, 1);

    tensor output_size(input_layout.size.batch[0], number_of_features,
                       output_range.spatial[0], output_range.spatial[1]);
    return { input_layout.data_type, input_layout.format, output_size };
}

std::string convolution_inst::to_string(convolution_node const& node)
{
    std::stringstream           primitive_description;
    auto desc                   = node.get_primitive();
    auto strd                   = desc->stride;
    auto weights_count          = node.weights(0).get_output_layout().count();
    auto bias_count             = node.bias_term() ? node.bias(0).get_output_layout().count() : 0;
    auto input                  = node.input();
    auto activation             = desc->with_activation ? " true" : "false";
    auto ud_out_size            = desc->with_output_size ? " true" : "false";

    primitive_description << "id: " << desc->id << ", type: convolution" << 
        "\n\tinput: " << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tweights count: " << weights_count << ", bias count: " << bias_count << 
        "\n\tstride: " << strd.spatial[0] << "x" << strd.spatial[1] << 
        "\n\twith activation: "<< activation <<", slope: "<< desc->activation_negative_slope <<
        "\n\twith user-defined out size: " << ud_out_size << ", dims: " << desc->output_size.spatial[0] << "x" << desc->output_size.spatial[1] <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ", size: " << node.get_output_layout().size << '\n';
    
    return primitive_description.str();
}

convolution_inst::typed_primitive_inst(network_impl& network, convolution_node const& node)
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
        auto& filter_inst = filter_mem.get_layout(); //convolution filter
        if (bias_term())
        {
            auto& bias_inst = bias_memory(j).get_layout();
            if (bias_inst.size.batch[0] != 1 && bias_inst.size.feature[0] != 1 && bias_inst.size.spatial[1] != 1)
                throw std::runtime_error("Biases isn't 1D vector."); // b=1, f=1
            if (bias_inst.size.spatial[0] != output_size.feature[0] / split)
                throw std::runtime_error("Biases/output feature maps number does not match.");
        }

        auto input_offset = argument.input_offset;

        if (filter_inst.size.raw.size() != output_inst.size.raw.size())
            throw std::runtime_error("Weights number of dimensions do not match output number of dimensions.");
        if (node.get_output_layout().data_padding.filling_value() != 0.0f)
            throw std::runtime_error("Unknown padding mode.");
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
