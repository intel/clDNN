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

#include "proposal_inst.h"
#include "primitive_type_base.h"

#include <cmath>

namespace cldnn
{

static void generate_anchors(unsigned base_size, const std::vector<float>& ratios, const std::vector<float>& scales,    // input
                             std::vector<proposal_inst::anchor>& anchors);                                              // output


primitive_type_id proposal_type_id()
{
    static primitive_type_base<proposal> instance;
    return &instance;
}


layout proposal_inst::calc_output_layout(proposal_node const& node)
{
    auto desc = node.get_primitive();
    layout input_layout = node.get_dependency(cls_scores_index).get_output_layout();

    return layout(input_layout.data_type, format::bfyx, { desc->post_nms_topn, 1, CLDNN_ROI_VECTOR_SIZE, 1 });
}

static inline std::string stringify_vector(std::vector<float> v)
{
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i)
    {
        s << v.at(i);
        if (i + 1 < v.size()) s << ", ";
    }

    s << " }";

    return s.str();
}

//TODO: rename to?
static std::string stringify_port(const program_node & p)
{
    std::stringstream res;

    res << "{ id: " << p.id()
        << ", layout: { count: " << p.get_output_layout().count()
        << ", size: " << p.get_output_layout().size << " } }";

    return res.str();
}


std::string proposal_inst::to_string(proposal_node const& node)
{
    std::stringstream                   primitive_description;
    auto desc                           = node.get_primitive();
    auto scales_parm                    = desc->scales;

    primitive_description
        << "{"
        << "\tid: " << desc->id << ", type: proposal\n"
        << "\tcls_score: " << stringify_port(node.cls_score()) << ",\n"
        << "\tbbox_pred: " << stringify_port(node.bbox_pred()) << ",\n"
        << "\timage_info: " << stringify_port(node.image_info()) << ",\n"
        << "\tparams: {\n"
        << "\t\tmax_proposals: " << desc->max_proposals << ",\n"
        << "\t\tiou_threshold: " << desc->iou_threshold << ",\n"
        << "\t\tmin_bbox_size: " << desc->min_bbox_size << ",\n"
        << "\t\tpre_nms_topn: " << desc->pre_nms_topn << ",\n"
        << "\t\tpost_nms_topn: " << desc->post_nms_topn << ",\n"
        << "\t\tratios: { " << stringify_vector(desc->ratios) << " },\n"
        << "\t\tscales: { " << stringify_vector(desc->scales) << " }\n"
        << "\t},\n"
        << "\toutput: {\n"
        << "\t\tpadding_lower_size: " << desc->output_padding.lower_size() << ",\n"
        << "\t\tpadding_upper_size: " << desc->output_padding.upper_size() << ",\n"
        << "\t\tcount: " << node.get_output_layout().count() << ",\n"
        << "\t\tsize: " << node.get_output_layout().size << "}\n"
        << "}\n";

    return primitive_description.str();
}

proposal_inst::typed_primitive_inst(network_impl& network, proposal_node const& node)
    :parent(network, node)
{
//    std::vector<float> default_ratios = { 0.5f, 1.0f, 2.0f };
    int default_size = 16;
    generate_anchors(default_size, argument.ratios, argument.scales, _anchors);
}

static void calc_basic_params(
        const proposal_inst::anchor& base_anchor,                       // input
        float& width, float& height, float& x_center, float& y_center)  // output
{
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}

static std::vector<proposal_inst::anchor> make_anchors(
        const std::vector<float>& ws,
        const std::vector<float>& hs,
        float x_center,
        float y_center)
{
    size_t len = ws.size();
    assert(hs.size() == len);

    std::vector<proposal_inst::anchor> anchors(len);

    for (size_t i = 0 ; i < len ; i++)
    {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }

    return anchors;
}

static std::vector<proposal_inst::anchor> calc_anchors(
        const proposal_inst::anchor& base_anchor,
        const std::vector<float>& scales)
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

    return make_anchors(ws, hs, x_center, y_center);
}

static std::vector<proposal_inst::anchor> calc_ratio_anchors(
        const proposal_inst::anchor& base_anchor,
        const std::vector<float>& ratios)
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

    return make_anchors(ws, hs, x_center, y_center);
}

static void generate_anchors(unsigned int base_size, const std::vector<float>& ratios, const std::vector<float>& scales,   // input
                     std::vector<proposal_inst::anchor>& anchors)                                                       // output
{
    float end = (float)(base_size - 1);        // because we start at zero

    proposal_inst::anchor base_anchor(0.0f, 0.0f, end, end);

    std::vector<proposal_inst::anchor> ratio_anchors = calc_ratio_anchors(base_anchor, ratios);

    for (auto& ratio_anchor : ratio_anchors)
    {
        std::vector<proposal_inst::anchor> tmp = calc_anchors(ratio_anchor, scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }
}
}
