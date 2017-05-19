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

#include "simpler_nms_inst.h"
#include "primitive_type_base.h"

#include <cmath>

namespace cldnn
{

static void generate_anchors(unsigned int base_size, const std::vector<float>& ratios, const std::vector<float>& scales,   // input
                             std::vector<simpler_nms_inst::anchor>& anchors);                                              // output


primitive_type_id simpler_nms_type_id()
{
    static primitive_type_base<simpler_nms, simpler_nms_inst> instance;
    return &instance;
}


layout simpler_nms_inst::calc_output_layout(simpler_nms_node const& node)
{
    auto desc = node.get_primitive();
    layout input_layout = node.get_dependency(cls_scores_index).get_output_layout();

    return layout(input_layout.data_type, format::bfyx, { desc->post_nms_topn, 1, CLDNN_ROI_VECTOR_SIZE, 1 });
}

std::string simpler_nms_inst::to_string(simpler_nms_node const& node)
{
    std::stringstream                   primitive_description;
    auto desc                           = node.get_primitive();
    auto scales_parm                    = desc->scales;
    std::stringstream                   ss_scales_param;
    for (size_t i = 0; i < scales_parm.size(); ++i)
    {
        ss_scales_param << scales_parm.at(i);
        i != (scales_parm.size() - 1) ? ss_scales_param << ", " : ss_scales_param << "";
    }  

    primitive_description << "id: " << desc->id << ", type: simpler_nms" << 
        "\n\tcls_scores id: " << node.cls_score().id()  << ", size: " << node.cls_score().get_output_layout().count() << ",  size: " << node.cls_score().get_output_layout().size <<
        "\n\tbbox_pred id:  " << node.bbox_pred().id()  << "\n" << node.bbox_pred().get_output_layout().count() << ",  size: " << node.bbox_pred().get_output_layout().size <<
        "\n\timage_info id: " << node.image_info().id() << "\n" << node.image_info().get_output_layout().count() << ",  size: " << node.image_info().get_output_layout().size <<
        "\n\tmax proposals: " << desc->max_proposals    << ", tiou_treshold: " << desc->iou_threshold << 
        "\n\tmin_bbox_size: " << desc->min_bbox_size    << ", feature_stride: " << desc->feature_stride <<
        "\n\tpre_nms_topn: "  << desc->pre_nms_topn     << ", post_nms_topn: " << desc->post_nms_topn <<
        "\n\tscales param: "  << ss_scales_param.str()  << 
        "\n\toutput padding lower size: " << desc->output_padding.lower_size() <<
        "\n\toutput padding upper size: " << desc->output_padding.upper_size() <<
        "\n\toutput: count: " << node.get_output_layout().count() << ",  size: " << node.get_output_layout().size << '\n';

    return primitive_description.str();
}

simpler_nms_inst::typed_primitive_inst(network_impl& network, simpler_nms_node const& node)
    :parent(network, node)
{
    std::vector<float> default_ratios = { 0.5f, 1.0f, 2.0f };
    int default_size = 16;
    generate_anchors(default_size, default_ratios, argument.scales, _anchors);
}

static void calc_basic_params(const simpler_nms_inst::anchor& base_anchor,                   // input
                            float& width, float& height, float& x_center, float& y_center)   // output
{
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}


static void make_anchors(const std::vector<float>& ws, const std::vector<float>& hs, float x_center, float y_center,   // input
                        std::vector<simpler_nms_inst::anchor>& anchors)                                                // output
{
    size_t len = ws.size();
    anchors.clear();
    anchors.resize(len);

    for (unsigned int i = 0 ; i < len ; i++)
    {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }
}


static void calc_anchors(const simpler_nms_inst::anchor& base_anchor, const std::vector<float>& scales,       // input
                        std::vector<simpler_nms_inst::anchor>& anchors)                                       // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    calc_basic_params(base_anchor, width, height, x_center, y_center);

    size_t num_scales = scales.size();
    std::vector<float> ws(num_scales), hs(num_scales);

    for (unsigned int i = 0 ; i < num_scales ; i++)
    {
        ws[i] = width * scales[i];
        hs[i] = height * scales[i];
    }

    make_anchors(ws, hs, x_center, y_center, anchors);
}


static void calc_ratio_anchors(const simpler_nms_inst::anchor& base_anchor, const std::vector<float>& ratios,      // input
                             std::vector<simpler_nms_inst::anchor>& ratio_anchors)                                 // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    calc_basic_params(base_anchor, width, height, x_center, y_center);

    float size = width * height;

    size_t num_ratios = ratios.size();

    std::vector<float> ws(num_ratios), hs(num_ratios);

    for (unsigned int i = 0 ; i < num_ratios ; i++)
    {
        float new_size = size / ratios[i];
        ws[i] = round(sqrt(new_size));
        hs[i] = round(ws[i] * ratios[i]);
    }

    make_anchors(ws, hs, x_center, y_center, ratio_anchors);
}

static void generate_anchors(unsigned int base_size, const std::vector<float>& ratios, const std::vector<float>& scales,   // input
                     std::vector<simpler_nms_inst::anchor>& anchors)                                                       // output
{
    float end = (float)(base_size - 1);        // because we start at zero

    simpler_nms_inst::anchor base_anchor(0.0f, 0.0f, end, end);

    std::vector<simpler_nms_inst::anchor> ratio_anchors;
    calc_ratio_anchors(base_anchor, ratios, ratio_anchors);

    std::vector<simpler_nms_inst::anchor> tmp_anchors;

    for (auto& ratio_anchor : ratio_anchors)
    {
        calc_anchors(ratio_anchor, scales, tmp_anchors);
        anchors.insert(anchors.end(), tmp_anchors.begin(), tmp_anchors.end());
        tmp_anchors.clear();
    }
}
}
