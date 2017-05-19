// Copyright (c) 2016-2017 Intel Corporation
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


#if FP16_SUPPORTED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif


KERNEL (concatenation_gpu_depth_yxfb)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint depth_offset)
{
    uint global_id = get_global_id(0);

    uint input_offset = global_id * INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM;
    uint output_offset = OUTPUT_BATCH_NUM * (depth_offset + global_id * OUTPUT_FEATURE_NUM);
    for(uint f = 0; f < INPUT_FEATURE_NUM * OUTPUT_BATCH_NUM; f++)
    {
        output[output_offset++] = input[input_offset++];
    }
}