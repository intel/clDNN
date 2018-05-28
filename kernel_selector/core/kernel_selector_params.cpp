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

    std::string Params::to_string() const
    {
        std::stringstream s;
        s << toString(kType);
        return s.str();
    }

    std::string BaseParams::to_string() const
    {
        std::stringstream s;
        s << Params::to_string() << "_";
        s << toString(activationParams) << "_";
        s << toString(activationFunc) << "_";

        for (auto input : inputs)
        {
            s << toString(input) << "_";
        }
        s << toString(output);

        return s.str();
    }

    std::string ConvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        if (bias.empty())
        {
            s << "no_bias" << "_";
        }
        else
        {
            s << "bias_" << bias[0].PhysicalSize() << "_";
        }
        s << convParams.filterSize.x << "_" << convParams.filterSize.y << "_";
        s << convParams.stride.x << "_" << convParams.stride.y << "_";
        s << convParams.dilation.x << "_" << convParams.dilation.y << "_";
        s << convParams.padding.x << "_" << convParams.padding.y << "_";
        s << convParams.split;

        return s.str();
    }

    std::string DeconvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        if (bias.empty())
        {
            s << "no_bias" << "_";
        }
        else
        {
            s << "bias_size:" << bias[0].PhysicalSize() << "_";
        }
        s << deconvParams.filterSize.x << "_" << deconvParams.filterSize.y << "_";
        s << deconvParams.stride.x << "_" << deconvParams.stride.y << "_";
        s << deconvParams.dilation.x << "_" << deconvParams.dilation.y << "_";
        s << deconvParams.padding.x << "_" << deconvParams.padding.y << "_";
        s << deconvParams.split;

        return s.str();
    }

    std::string ConvolutionGradWeightsParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        if (bias.empty())
        {
            s << "no_bias" << "_";
        }
        else
        {
            s << "bias_" << bias[0].PhysicalSize() << "_";
        }
        s << convGradWeightsParams.filterSize.x << "_" << convGradWeightsParams.filterSize.y << "_";
        s << convGradWeightsParams.stride.x << "_" << convGradWeightsParams.stride.y << "_";
        s << convGradWeightsParams.dilation.x << "_" << convGradWeightsParams.dilation.y << "_";
        s << convGradWeightsParams.padding.x << "_" << convGradWeightsParams.padding.y << "_";
        s << convGradWeightsParams.split;

        return s.str();
    }
}