// Copyright (c) 2018 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "pyramidROIAlign_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn {
    primitive_type_id PyramidROIAlign_type_id()
    {
        static primitive_type_base<PyramidROIAlign> instance;
        return &instance;
    }

    layout pyramidROIAlign_inst::calc_output_layout(pyramidROIAlign_node const &node)
    {

        auto desc = node.get_primitive();

        auto boxes_layout = node.boxes().get_output_layout();
        auto P2_layout = node.P2().get_output_layout();
        auto pool_size_layout = node.pool_size().get_output_layout();
 
        int32_t output_b = boxes_layout.size.spatial[1];
        int32_t output_f = P2_layout.size.feature[0];

        int32_t output_x = pool_size_layout.size.spatial[0];
        int32_t output_y = pool_size_layout.size.spatial[1];
       
        return layout{ P2_layout.data_type, P2_layout.format, { output_b, output_f, output_x, output_y } };
    }

    std::string pyramidROIAlign_inst::to_string(pyramidROIAlign_node const& node)
    {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();
        std::stringstream primitive_description;
        json_composite pyramidROIAlign_info;
        node_info->add("pyramidROIAlign_info", pyramidROIAlign_info);
        node_info->dump(primitive_description);
        return primitive_description.str();
    }
    
    pyramidROIAlign_inst::typed_primitive_inst(network_impl& network, pyramidROIAlign_node const& node)
        : parent(network, node)
    { }
}
