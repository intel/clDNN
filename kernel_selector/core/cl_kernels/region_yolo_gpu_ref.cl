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

#include "include/include_all.cl"

#define IW INPUT0_SIZES[0]
#define IH INPUT0_SIZES[1]
#define IC INPUT0_SIZES[2]
#define IB INPUT0_SIZES[3]

#define INPUTS_COUNT (IH * IW * NUM * (CLASSES + COORDS + 1))

inline UNIT_TYPE logistic_activate(UNIT_TYPE x) {
    return 1. / (1. + exp(-x));
}

inline int entry_index(int width, int height, int coords, int classes,
                       int outputs, int batch, int location,
                       int entry) {
    int n = location / (width * height);
    int loc = location % (width * height);
    return batch * outputs + n * width * height * (coords + classes + 1) +
        entry * width * height + loc;
}

static
void softmax_generic(const __global UNIT_TYPE* src_data, __global UNIT_TYPE* dst_data,
                     int B, int C, int W, int H, int h, int w)
{
    int i = h * W + w;
    for (int b = 0; b < B; b++) {
        UNIT_TYPE max = src_data[b*C*H*W + i];
        for (int c = 0; c < C; c++) {
            UNIT_TYPE val = src_data[b*C*H*W + c*H*W + i];
            if (val > max) max = val;
        }

        UNIT_TYPE expSum = 0;
        for (int c = 0; c < C; c++) {
            dst_data[b*C*H*W + c*H*W + i] = exp(src_data[b*C*H*W + c*H*W + i] - max);
            expSum += dst_data[b*C*H*W + c*H*W + i];
        }

        for (int c = 0; c < C; c++) {
            dst_data[b*C*H*W + c*H*W + i] = dst_data[b*C*H*W + c*H*W + i] / expSum;
        }
    }
}

KERNEL (region_yolo_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    int w = get_global_id(0);
    int h = get_global_id(1);

    for (int b = 0; b < IB; b++) {
        for (int n = 0; n < NUM; n++) {
            // coords: x/y
            int index = entry_index(IW, IH, COORDS, CLASSES, INPUTS_COUNT, b, n * IW * IH, 0);
            int i = index + 2 * IW * h + 2 * w;
            output[i] = logistic_activate(input[i]);
            output[i+1] = logistic_activate(input[i+1]);

            // coords: w/h: directly copy?
            index = entry_index(IW, IH, COORDS, CLASSES, INPUTS_COUNT, b, n * IW * IH, 2);
            i = index + 2 * IW * h + 2 * w;
            output[i] = input[i];
            output[i+1] = input[i+1];

            // confidence
            index = entry_index(IW, IH, COORDS, CLASSES, INPUTS_COUNT, b, n * IW * IH, COORDS);
            i = index + IW * h + w;
            output[i] = logistic_activate(input[i]);
        }
    }

    // the probability of classes (20)
    int index = entry_index(IW, IH, COORDS, CLASSES, INPUTS_COUNT, 0, 0, COORDS + 1);
    int batch_offset = INPUTS_COUNT / NUM;
    for (int b = 0; b < IB * NUM; b++)
        softmax_generic(input + index + b * batch_offset, output + index + b * batch_offset,
                        1, CLASSES, IH, IW, h, w);
}
