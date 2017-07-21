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

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include <sstream>
 
namespace KernelSelector {

    std::string BaseParams::to_string() const
    {
        std::stringstream s;
        s << toString(inputs[0].GetDType()) << "_";
        s << toString(inputs[0].GetLayout()) << "_";
        s << toString(output.GetLayout()) << "_";
        s << toString(activationFunc) << "_";
        s << nlParams.m << "_" << nlParams.n << "_";
        s << inputs[0].X().v << "_" << inputs[0].Y().v << "_" << inputs[0].Feature().v << "_" << inputs[0].Batch().v << "_";
        //s << inputs[0].offset << "_" << inputs[0].x().pitch << "_" << inputs[0].y().pitch << "_" << inputs[0].feature().pitch << "_" << inputs[0].batch().pitch << "_";
        s << output.X().v << "_" << output.Y().v << "_" << output.Feature().v << "_" << output.Batch().v;
        //s << output.offset << "_" << output.x().pitch << "_" << output.y().pitch << "_" << output.feature().pitch << "_" << output.batch().pitch;
        return s.str();
    }

    std::string ConvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        s << toString(weights.GetLayout()) << "_";
        if (bias.size())
        {
            s << toString(bias[0].GetLayout()) << "_";
        }
        else
        {
            s << "nobias_";
        }
        s << convParams.filterSize.x << "_" << convParams.filterSize.y << "_";
        s << convParams.padding.x << "_" << convParams.padding.y << "_";
        s << convParams.stride.x << "_" << convParams.stride.y << "_";
        s << convParams.dilation.x << "_" << convParams.dilation.y;

        return s.str();
    }

    std::string DeconvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        s << toString(weights.GetLayout()) << "_";
        if (bias.size())
        {
            s << toString(bias[0].GetLayout()) << "_";
        }
        else
        {
            s << "nobias_";
        }
        s << deconvParams.filterSize.x << "_" << deconvParams.filterSize.y << "_";
        s << deconvParams.padding.x << "_" << deconvParams.padding.y << "_";
        s << deconvParams.stride.x << "_" << deconvParams.stride.y << "_";
        s << deconvParams.dilation.x << "_" << deconvParams.dilation.y;

        return s.str();
    }
}