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

KERNEL(reorder_weights_winograd_2x3_s1)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const int input_tile_width = 3;
    const int input_tile_height = 1;
    const int output_tile_width = 4;
    const int output_tile_height = 1;
    
    const int tile_x_idx = get_global_id(0);
    const int tile_y_idx = get_global_id(1);
    const int feature_idx = get_global_id(2) % INPUT0_IFM_NUM;
    const int batch_idx = get_global_id(2) / INPUT0_IFM_NUM;
    
    int in_idx = batch_idx * INPUT0_OFM_PITCH
                 + feature_idx * INPUT0_IFM_PITCH
                 + tile_y_idx * input_tile_height * INPUT0_Y_PITCH
                 + tile_x_idx * input_tile_width * INPUT0_X_PITCH;

    INPUT0_TYPE tile[3];
    tile[0] = input[in_idx]; in_idx += INPUT0_X_PITCH;
    tile[1] = input[in_idx]; in_idx += INPUT0_X_PITCH;
    tile[2] = input[in_idx];


    int out_idx = batch_idx * OUTPUT_OFM_PITCH
                  + feature_idx * OUTPUT_IFM_PITCH
                  + tile_y_idx * output_tile_height * OUTPUT_Y_PITCH
                  + tile_x_idx * output_tile_width * OUTPUT_X_PITCH;


    output[out_idx] = tile[0]; out_idx += OUTPUT_X_PITCH;
    output[out_idx] = (tile[0] + tile[1] + tile[2]) / 2.0f; out_idx += OUTPUT_X_PITCH;
    output[out_idx] = (tile[0] - tile[1] + tile[2]) / 2.0f; out_idx += OUTPUT_X_PITCH;
    output[out_idx] = tile[2];
}
