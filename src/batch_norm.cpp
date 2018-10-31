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
#include "json_object.h"
#include "mutable_data_inst.h"

namespace cldnn
{
primitive_type_id batch_norm_type_id()
{
    static primitive_type_base<batch_norm> instance;
    return &instance;
}

layout batch_norm_inst::calc_output_layout(batch_norm_node const& node)
{
    return node.input().get_non_padded_output_layout();
}

std::string batch_norm_inst::to_string(batch_norm_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& mean     = node.mean();
	auto& scale = node.scale();
	auto& shift = node.shift();
    bool variance_term = node.variance_term();
    auto& inv_var  = node.inv_variance();

    std::stringstream primitive_description;

    json_composite batch_norm_info;
    if (node.use_global_stats())
    {
        batch_norm_info.add("mean_id", mean.id());

        if (variance_term)
        {
            batch_norm_info.add("variance_id", node.variance().id());
        }
    }
	if (node.use_scale_shift())
	{
		batch_norm_info.add("scale_id", scale.id());
		batch_norm_info.add("shift_id", shift.id());
	}
    if (node.forwad_pass())
    {
        batch_norm_info.add("inv_var", inv_var.id());
    }
    batch_norm_info.add("epsilon", desc->epsilon);

    node_info->add("batch norm info", batch_norm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

batch_norm_inst::typed_primitive_inst(network_impl& network, batch_norm_node const& node)
    :parent(network, node) 
{
    if (use_global_stats())
    {
        auto mean_format = node.mean().get_output_layout().format;
        auto variance_format = node.variance().get_output_layout().format;

        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Mean format", mean_format.value, "supported mean formats", format::yxfb, format::bfyx, format::byxf);
        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Variance format", variance_format.value, "supported variance formats", format::yxfb, format::bfyx, format::byxf);

		auto is_mean_mutable_data = node.mean().is_type<mutable_data>();
		auto is_var_mutable_data = node.variance().is_type<mutable_data>();

		CLDNN_ERROR_BOOL(node.id(), "mean and variance are not the same type", (is_mean_mutable_data != is_var_mutable_data), "");
    }

	if (use_scale_shift()) {
		auto scale_format = node.scale().get_output_layout().format;
		auto shift_format = node.shift().get_output_layout().format;

		CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Scale format", scale_format.value, "supported scale formats", format::yxfb, format::bfyx, format::byxf);
		CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Shift format", shift_format.value, "supported shift formats", format::yxfb, format::bfyx, format::byxf);
	}

	if (forwad_pass())
	{
		auto is_inv_var_mutable_data = node.inv_variance().is_type<mutable_data>();
		CLDNN_ERROR_BOOL(node.id(), "inv_variance is not mutable_data type", !is_inv_var_mutable_data, "");
	}
}
}