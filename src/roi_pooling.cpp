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

#include "roi_pooling_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"

namespace cldnn
{
primitive_type_id roi_pooling_type_id()
{
    static primitive_type_base<roi_pooling> instance;
    return &instance;
}

layout roi_pooling_inst::calc_output_layout(roi_pooling_node const& node)
{
    auto desc = node.get_primitive();
    layout data_layout = node.input().get_output_layout();
    int fm = data_layout.size.feature[0];

    layout rois_layout = node.rois().get_output_layout();
    int num_rois = rois_layout.size.batch[0];

    int gss = desc->group_sz * desc->group_sz;


    CLDNN_ERROR_LESS_THAN(node.id(), "Group size", desc->group_sz, "value", 0, "");
    if (gss && fm % gss != 0)
    {
        CLDNN_ERROR_MESSAGE(node.id(), "group_sz must be either 0 (For RoIPooling) or satisfy fm % (group_sz^2) == 0");
    }
    
    if (gss)
    {
        fm /= gss;
    }

    return layout(rois_layout.data_type, format::bfyx, { num_rois, fm, desc->pooled_width, desc->pooled_height });
}

std::string roi_pooling_inst::to_string(roi_pooling_node const& node)
{
    std::stringstream primitive_description;
    auto desc           = node.get_primitive();
    auto mode           = desc->mode == pooling_mode::max ? "max" : "average";

    primitive_description
        << "{\n"
        << "\tid: " << desc->id << ",\n"
        << "\ttype: roi_pooling,\n"
        << "\tparams: {\n"
        << "\t\tmode: " << mode << ",\n"
        << "\t\tpooled_w: " << desc->pooled_width << ",\n"
        << "\t\tpooled_h: " << desc->pooled_height << ",\n"
        << "\t\tspatial_scale: " << desc->spatial_scale << ",\n"
        << "\t\tgroup_sz: " << desc->group_sz << "\n"
        << "\t}\n"
        << "\toutput: {\n"
        << "\t\tpad_lower_sz: " << desc->output_padding.lower_size() << ",\n"
        << "\t\tpad_upper_sz: " << desc->output_padding.upper_size() << ",\n"
        << "\t\tcount: " << node.get_output_layout().count() << ",\n"
        << "\t\tsize: " << node.get_output_layout().size << "\n"
        << "\t}\n"
        << "}\n";

    return primitive_description.str();
}

}
