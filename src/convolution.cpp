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
#include "error_handler.h"

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
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "Stride spatial X", stride.spatial[0], "value", 0, "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "Stride spatial Y", stride.spatial[1], "value", 0, "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "Dilatation spatial X", dilation.spatial[0], "value", 0, "Dilatation patial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "Dilatation spatial Y", dilation.spatial[1], "value", 0, "Dilatation spatial Y must be positive (>= 1)");
    CLDNN_ERROR_GREATER_THAN(node.id(), "Input offset spatial X", 2 * input_offset.spatial[0], "input layout spatial X", input_layout.size.spatial[0], "There is no input data to process");
    CLDNN_ERROR_GREATER_THAN(node.id(), "Input offset spatial Y", 2 * input_offset.spatial[1], "input layout spatial Y", input_layout.size.spatial[1], "There is no input data to process");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input offset feature", input_offset.feature[0], "", 0, "Input offset in feature is not supported");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input offset batch", input_offset.batch[0], "", 0, "Input offset in batch is not supported");

    // TODO: FCN and SSD used offset larger than convolution size. does it make sense to support it? do we support it on the ref kernels?
//     CLDNN_ERROR_GREATER_THAN(node.id(), "Negate input offset spatial X", -input_offset.spatial[0], "input window size spatial X", filter_size.spatial[0], "First convolution is outside of image. please reduce input offset X");
//     CLDNN_ERROR_GREATER_THAN(node.id(), "Negate input offset spatial Y", -input_offset.spatial[1], "input window size spatial Y", filter_size.spatial[1], "First convolution is outside of image. please reduce input offset Y");

    // get output feature map from weights. It should be the same as number of biases. Will be verifed in convolution::create()
    auto number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(split);

    if (desc->with_output_size)
    {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "User defined output spatial X", desc->output_size.spatial[0], "value", 0, "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(), "User defined output spatial Y", desc->output_size.spatial[1], "value", 0, "must be positive(>= 1)");

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
    auto& input                 = node.input();
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

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input number of dimensions", input_inst.size.raw.size(), "output number of dimensions", output_inst.size.raw.size(), "Input/output dims mismtach");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Stride number of dimensions", stride.raw.size(), "output number of dimensions", output_inst.size.raw.size(), "stride/output dims mismtach");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++)
    {
        auto& filter_mem = weights_memory(j);
        auto& filter_inst = filter_mem.get_layout(); //convolution filter
        if (bias_term())
        {
            auto& bias_inst = bias_memory(j).get_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Bias batch[0]", bias_inst.size.batch[0], "expected size of batch", 1, "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Bias feature[0]", bias_inst.size.feature[0], "expected size of feature", 1, "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Bias spatial[1]", bias_inst.size.spatial[1], "expected size of spatial[1]", 1, "Biases isn't 1D vector.");
          
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Bias spatial[0]", bias_inst.size.spatial[0], "expected feature map number", output_size.feature[0] / split, "Bias/fm mismtach");
        }

        auto input_offset = argument.input_offset;

        CLDNN_ERROR_NOT_EQUAL(node.id(), "Weights number of dimensions", filter_inst.size.raw.size(), "output number of dimensions", output_inst.size.raw.size(), "Weights/output dims mismtach");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Convolution padding mode", node.get_output_layout().data_padding.filling_value(), "padding value", 0.0f, "Unknown padding mode.");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Input offset number of dimensions", input_offset.raw.size(), "input number of dimensions", input_inst.size.raw.size(), "Input offset/ input size mismtach");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Output feature size", output_size.feature.size(), "expected feature size", 1, "Only one-dimensional features are supported");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Output batch size", output_size.batch.size(), "expected output size", 1, "Only one-dimensional batch size are supported");
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Weights spatial size", filter_inst.size.spatial.size(), "expected weights spatial size", 2, "Weights have to have 2 dimensions in spatial domain.");
        CLDNN_ERROR_LESS_THAN(node.id(), "Weights feature maps number", (input_inst.size.feature[0] - input_offset.feature[0]) / split, "input feature maps number", filter_inst.size.feature[0], "Weights/ifm mismtach");
    }
}
}
