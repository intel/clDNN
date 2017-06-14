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

#include "prior_box_inst.h"
#include "primitive_type_base.h"

#include <cmath>

namespace cldnn
{
primitive_type_id prior_box_type_id()
{
    static primitive_type_base<prior_box> instance;
    return &instance;
}

layout prior_box_inst::calc_output_layout(prior_box_node const& node)
{
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();
    assert(input_layout.size.spatial.size() == 2);

    const int layer_width = input_layout.size.spatial[0];
    const int layer_height = input_layout.size.spatial[1];

    int num_priors = (int)desc->aspect_ratios.size() * (int)desc->min_sizes.size() + (int)desc->max_sizes.size();

    // Since all images in a batch has same height and width, we only need to
    // generate one set of priors which can be shared across all images.
    // 2 features. First feature stores the mean of each prior coordinate.
    // Second feature stores the variance of each prior coordinate.

    return{ input_layout.data_type, cldnn::format::bfyx, cldnn::tensor( 1, 2, 1, layer_width * layer_height * num_priors * 4 ) };
}

std::string vector_to_string(std::vector<float> vec)
{
    std::stringstream result;
    for (size_t i = 0; i < vec.size(); i++)
        result << vec.at(i) << ", ";
    return result.str();
}

std::string prior_box_inst::to_string(prior_box_node const& node)
{
    std::stringstream               primitive_description;
    auto desc                       = node.get_primitive();
    auto input                      = node.input();
    auto flip                       = desc->flip ? "true" : "false";
    auto clip                       = desc->clip ? "true" : "false";
    std::string str_min_sizes       = vector_to_string(desc->min_sizes);
    std::string str_max_sizes       = vector_to_string(desc->max_sizes);
    std::string str_aspect_ratio    = vector_to_string(desc->aspect_ratios);
    std::string str_variance        = vector_to_string(desc->variance);

    primitive_description << "id: " << desc->id << ", type: normalization" <<
        "\n\tinput: "        << input.id() << ", count: " << input.get_output_layout().count() << ", size: " << input.get_output_layout().size <<
        "\n\timage size: "   << desc->img_size <<
        "\n\tmin_sizes: "    << str_min_sizes << ", max sizes: " << str_max_sizes <<
        "\n\taspect_ratio: " << str_aspect_ratio <<
        "\n\tflip: "         << flip << "clip: " << clip <<
        "\n\tstep_width: "   << desc->step_width << ", step_height: " << desc->step_height << ", offset: " << desc->offset <<
        "\n\tstr_variance: " << str_variance <<
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

template<typename dtype>
void prior_box_inst::generate_output()
{
    // Calculate output.
    // All the inputs for this layer are known at this point,
    // so the output buffer is written here and not in execute().
    const auto& input_mem = input_memory();
    const auto& output_mem = output_memory();

    const int layer_width = input_mem.get_layout().size.spatial[0];
    const int layer_height = input_mem.get_layout().size.spatial[1];
    const int img_width = argument.img_size.spatial[0];
    const int img_height = argument.img_size.spatial[1];
    float step_w = argument.step_width;
    float step_h = argument.step_height;
    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }
    const float offset = argument.offset;
    int num_priors = (int)argument.aspect_ratios.size() * (int)argument.min_sizes.size() + (int)argument.max_sizes.size();

    auto out_ptr = output_mem.pointer<dtype>();
    int dim = layer_height * layer_width * num_priors * 4;
    int idx = 0;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width, box_height;
            for (size_t s = 0; s < argument.min_sizes.size(); ++s) {
                float min_size = argument.min_sizes[s];
                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size;
                // xmin
                out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                // ymin
                out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                // xmax
                out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                // ymax
                out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);

                if (argument.max_sizes.size() > 0) {
                    float max_size_ = argument.max_sizes[s];
                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = sqrt(min_size * max_size_);
                    // xmin
                    out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                    // ymin
                    out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                    // xmax
                    out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                    // ymax
                    out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);
                }

                // rest of priors
                for (size_t r = 0; r < argument.aspect_ratios.size(); ++r) {
                    float ar = argument.aspect_ratios[r];
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }
                    box_width = min_size * sqrt(ar);
                    box_height = min_size / sqrt(ar);
                    // xmin
                    out_ptr[idx++] = (dtype)((center_x - box_width / 2.f) / img_width);
                    // ymin
                    out_ptr[idx++] = (dtype)((center_y - box_height / 2.f) / img_height);
                    // xmax
                    out_ptr[idx++] = (dtype)((center_x + box_width / 2.f) / img_width);
                    // ymax
                    out_ptr[idx++] = (dtype)((center_y + box_height / 2.f) / img_height);
                }
            }
        }
    }

    // clip the prior's coordinate such that it is within [0, 1]
    if (argument.clip) {
        for (int d = 0; d < dim; ++d) {
            out_ptr[d] = (dtype)std::min(std::max((float)out_ptr[d], 0.f), 1.f);
        }
    }

    // set the variance.
    int count = output_mem.get_layout().size.spatial[0] * output_mem.get_layout().size.spatial[1];
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            for (int i = 0; i < num_priors; ++i) {
                for (int j = 0; j < 4; ++j) {
                    out_ptr[count] = (dtype)((argument.variance.size() == 1) ? argument.variance[0] : argument.variance[j]);
                    ++count;
                }
            }
        }
    }
}

prior_box_inst::typed_primitive_inst(network_impl& network, prior_box_node const& node)
    :parent(network, node)
{
    //Check arguments
    if (argument.min_sizes.size() == 0) {
        throw std::runtime_error("Must provide at least one min size.");
    }
    for (size_t i = 0; i < argument.min_sizes.size(); i++) {
        if (argument.min_sizes[i] <= 0) {
            throw std::runtime_error("Min size must be positive.");
        }
    }
    if ( (argument.max_sizes.size() > 0) && (argument.min_sizes.size() != argument.max_sizes.size())) {
        throw std::runtime_error("Number of min sizes must be equal to number of max sizes.");
    }
    for (size_t i = 0; i < argument.max_sizes.size(); i++) {
        if (argument.min_sizes[i] >= argument.max_sizes[i]) {
            throw std::runtime_error("Max size must be greater than Min size.");
        }
    }
    if (argument.variance.size() > 1) {
        if (argument.variance.size() != 4) {
            throw std::runtime_error("Must provide 4 variances.");
        }
        for (size_t i = 0; i < argument.variance.size(); i++) {
            if (argument.variance[i] <= 0) {
                throw std::runtime_error("Variance must be positive.");
            }
        }
    }
    else if (argument.variance.size() == 1) {
        if (argument.variance[0] <= 0) {
            throw std::runtime_error("Variance must be positive.");
        }
    }
    if ((argument.img_size.spatial[0] <= 0) || (argument.img_size.spatial[1] <= 0)) {
        throw std::runtime_error("Image dimensions must be positive.");
    }
    if ((argument.step_height < 0) || (argument.step_width < 0)) {
        throw std::runtime_error("Step dimensions must be positive.");
    }

    if (node.is_padded())
    {
        throw std::invalid_argument("Prior-box layer doesn't support output padding.");
    }

    if (input_memory().get_layout().data_type == data_types::f32)
    {
        generate_output<data_type_to_type<data_types::f32>::type>();
    }
    else
    {
        generate_output<data_type_to_type<data_types::f16>::type>();
    }
}
}
