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


#if FP16_UNIT_USED
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

KERNEL (crop_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint batch_id = get_global_id(0);
    const uint feature_id = get_global_id(1);
    const uint x = get_global_id(2) % OUTPUT_SIZE_X;
    const uint y = get_global_id(2) / OUTPUT_SIZE_X;
#if CROP_BFYX_USED
    const uint output_linear_id = x + OUTPUT_SIZE_X * (y + OUTPUT_SIZE_Y * (feature_id + OUTPUT_FEATURE_NUM * batch_id));
    const uint linear_id = (x + OFFSETS_SIZE_X) + INPUT_SIZE_X * ((y + OFFSETS_SIZE_Y) + INPUT_SIZE_Y * ((feature_id + OFFSETS_FEATURE_NUM) + INPUT_FEATURE_NUM * (batch_id + OFFSETS_BATCH_NUM)));
#else
    const uint output_linear_id = batch_id + OUTPUT_BATCH_NUM * (feature_id + OUTPUT_FEATURE_NUM * (x + OUTPUT_SIZE_X * y));
    const uint linear_id = (batch_id + OFFSETS_BATCH_NUM) + INPUT_BATCH_NUM * ((feature_id + OFFSETS_FEATURE_NUM) + INPUT_FEATURE_NUM * ((x + OFFSETS_SIZE_X) + INPUT_SIZE_X * (y + OFFSETS_SIZE_Y)));
#endif

    output[output_linear_id] = input[linear_id];
}