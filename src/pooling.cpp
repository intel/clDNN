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

#include "pooling_inst.h"
#include "primitive_type_base.h"


namespace cldnn
{
primitive_type_id pooling_type_id()
{
    static primitive_type_base<pooling, pooling_inst> instance;
    return &instance;
}

layout pooling_inst::calc_output_layout(parent::typed_node const& node)
{
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto input_spatial_size = node.input().get_output_layout().size.spatial.size();

    if (input_spatial_size != 2)
        throw std::runtime_error("Only two dimensional spatials are supported by pooling");

    auto input_offsets = desc->input_offset.sizes();
    auto strides = desc->stride.sizes();
    auto window_sizes = desc->size.sizes();
    //TODO !!!implement correct output size calculation!!!
    auto output_sizes = input_layout.size.sizes();
    auto spatial_offset = CLDNN_TENSOR_BATCH_DIM_MAX + CLDNN_TENSOR_FEATURE_DIM_MAX;

    for (decltype(input_spatial_size) i = spatial_offset; i < input_spatial_size + spatial_offset; i++)
    {
            // TODO: Consider moving general parameter verification to arguments constructor.
            if (strides[i] <= 0)
                throw std::runtime_error("Stride must be positive (>= 1)");
            if (2 * input_offsets[i] >= output_sizes[i])
                throw std::runtime_error("Input offset is greater than input data range. There is no input data to process");

            output_sizes[i] = static_cast<cldnn::tensor::value_type>(
                2 * input_offsets[i] < output_sizes[i]
                // ? std::max(output_sizes[i] - 2 * input_offsets[i] - window_sizes[i], 0) / strides[i] + 1
                ? ceil_div(std::max(output_sizes[i] - 2 * input_offsets[i] - window_sizes[i], 0), strides[i]) + 1
                : 0);
    }

    return{ input_layout.data_type, input_layout.format, output_sizes };
}

std::string pooling_inst::to_string(pooling_node const& node)
{
    std::stringstream   primitive_description;
    auto desc           = node.get_primitive();
    auto input          = node.input();
    auto strd           = desc->stride;
    auto kernel_size    = desc->size;
    auto mode           = desc->mode == pooling_mode::average ? "avarage" : "max";   

    primitive_description << "id: " << desc->id << ", type: pooling" << ", mode: " << mode <<
        "\n\tinput: "         << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\tstride: "        << strd.spatial[0] << "x" << strd.spatial[1] << 
        "\n\tkernel size: "   << kernel_size.spatial[0] << "x" << kernel_size.spatial[1] << 
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';
    
    return primitive_description.str();
}

}
