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

#include "convolution_kernel_selector.h"
#include "convolution_kernel_ref.h"
#include "convolution_kernel_gemm_like.h"
#include "convolution_kernel_direct_10_12_16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "convolution_kernel_yxfb_ref.h"
#include "convolution_kernel_yxfb_yxio_b16.h"
#include "convolution_kernel_yxfb_yxio_b8.h"
#include "convolution_kernel_yxfb_yxio_b1_block.h"
#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
#include "convolution_kernel_tutorial1.h"
#include <iostream>
 
namespace KernelSelector 
{
    ConvolutionKernelSelctor::ConvolutionKernelSelctor()
    {
        Attach<ConvolutionKernelRef>();
        Attach<ConvolutionKernelGEMMLike>();
        Attach<ConvolutionKernelDirect_10_10_12>();
        Attach<ConvolutionKernel_bfyx_os_iyx_osv16>();
        Attach<ConvolutionKernel_yxfb_Ref>();
        Attach<ConvolutionKernel_yxfb_yxio_b16>();
        Attach<ConvolutionKernel_yxfb_yxio_b8>();
        //Attach<ConvolutionKernel_yxfb_yxio_b1_block>(); // TODO: need to finish integration
        Attach<ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x>();
        Attach<ConvolutionKernel_Tutorial1>();
    }

    KernelsData ConvolutionKernelSelctor::GetBestKernels(const Params& params, const OptionalParams& options) const
    {
        //const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        //std::cout << orgParams.to_string() << std::endl;
        return GetNaiveBestKernel(params, options, KernelType::CONVOLUTION);
    }
}