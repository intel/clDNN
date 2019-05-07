/*****************************************************************************\

Copyright 2016 Intel Corporation All Rights Reserved.

The source code contained or described herein and all documents related to
the source code ("Material") are owned by Intel Corporation or its suppliers
or licensors. Title to the Material remains with Intel Corporation or its
suppliers and licensors. The Material contains trade secrets and proprietary
and confidential information of Intel or its suppliers and licensors. The
Material is protected by worldwide copyright and trade secret laws and
treaty provisions. No part of the Material may be used, copied, reproduced,
modified, published, uploaded, posted, transmitted, distributed, or
disclosed in any way without Intel's prior express written permission.

No license under any patent, copyright, trade secret or other intellectual
property right is granted to or conferred upon you by disclosure or delivery
of the Materials, either expressly, by implication, inducement, estoppel or
otherwise. Any license under such intellectual property rights must be
express and approved by Intel in writing.

File Name: cl_intel_device_side_avc_motion_estimation.h

Abstract:
    Host-side enumerations for cl_intel_device_side_avc_motion_estimation extension.

Notes:
    - Create directory %INTELOCLSDKROOT%\include\CL\IntelNDA
    - Copy file to %INTELOCLSDKROOT%\include\CL\IntelNDA.
    - In OpenCL host application add "#include <CL/IntelNDA/cl_device_side_avc_me.h>"

\*****************************************************************************/
#ifndef __CL_INTEL_DEVICE_SIDE_AVC_MOTION_ESTIMATION_H
#define __CL_INTEL_DEVICE_SIDE_AVC_MOTION_ESTIMATION_H

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************
* cl_intel_device_side_avc_motion_estimation extension *
********************************************************/
  
#define CL_DEVICE_AVC_ME_VERSION_INTEL                           0x410B
#define CL_DEVICE_AVC_ME_SUPPORTS_TEXTURE_SAMPLER_USE_INTEL      0x410C
#define CL_DEVICE_AVC_ME_SUPPORTS_PREEMPTION_INTEL               0x410D

#define CL_AVC_ME_VERSION_0_INTEL                         0x0;  // No support.
#define CL_AVC_ME_VERSION_1_INTEL                         0x1;  // First supported version.

#define CL_AVC_ME_MAJOR_16x16_INTEL                       0x0
#define CL_AVC_ME_MAJOR_16x8_INTEL                        0x1
#define CL_AVC_ME_MAJOR_8x16_INTEL                        0x2
#define CL_AVC_ME_MAJOR_8x8_INTEL                         0x3

#define CL_AVC_ME_MINOR_8x8_INTEL                         0x0
#define CL_AVC_ME_MINOR_8x4_INTEL                         0x1
#define CL_AVC_ME_MINOR_4x8_INTEL                         0x2
#define CL_AVC_ME_MINOR_4x4_INTEL                         0x3

#define CL_AVC_ME_MAJOR_FORWARD_INTEL                     0x0
#define CL_AVC_ME_MAJOR_BACKWARD_INTEL                    0x1
#define CL_AVC_ME_MAJOR_BIDIRECTIONAL_INTEL               0x2

#define CL_AVC_ME_PARTITION_MASK_ALL_INTEL                0x0
#define CL_AVC_ME_PARTITION_MASK_16x16_INTEL              0x7E
#define CL_AVC_ME_PARTITION_MASK_16x8_INTEL               0x7D
#define CL_AVC_ME_PARTITION_MASK_8x16_INTEL               0x7B
#define CL_AVC_ME_PARTITION_MASK_8x8_INTEL                0x77
#define CL_AVC_ME_PARTITION_MASK_8x4_INTEL                0x6F
#define CL_AVC_ME_PARTITION_MASK_4x8_INTEL                0x5F
#define CL_AVC_ME_PARTITION_MASK_4x4_INTEL                0x3F

#define CL_AVC_ME_SEARCH_WINDOW_EXHAUSTIVE_INTEL          0x0
#define CL_AVC_ME_SEARCH_WINDOW_SMALL_INTEL               0x1
#define CL_AVC_ME_SEARCH_WINDOW_TINY_INTEL                0x2
#define CL_AVC_ME_SEARCH_WINDOW_EXTRA_TINY_INTEL          0x3
#define CL_AVC_ME_SEARCH_WINDOW_DIAMOND_INTEL             0x4
#define CL_AVC_ME_SEARCH_WINDOW_LARGE_DIAMOND_INTEL       0x5
#define CL_AVC_ME_SEARCH_WINDOW_RESERVED0_INTEL           0x6
#define CL_AVC_ME_SEARCH_WINDOW_RESERVED1_INTEL           0x7
#define CL_AVC_ME_SEARCH_WINDOW_CUSTOM_INTEL              0x8
#define CL_AVC_ME_SEARCH_WINDOW_16x12_RADIUS_INTEL        0x9
#define CL_AVC_ME_SEARCH_WINDOW_4x4_RADIUS_INTEL          0x2
#define CL_AVC_ME_SEARCH_WINDOW_2x2_RADIUS_INTEL          0xa

#define CL_AVC_ME_SAD_ADJUST_MODE_NONE_INTEL              0x0
#define CL_AVC_ME_SAD_ADJUST_MODE_HAAR_INTEL              0x2

#define CL_AVC_ME_SUBPIXEL_MODE_INTEGER_INTEL             0x0
#define CL_AVC_ME_SUBPIXEL_MODE_HPEL_INTEL                0x1
#define CL_AVC_ME_SUBPIXEL_MODE_QPEL_INTEL                0x3

#define CL_AVC_ME_COST_PRECISION_QPEL_INTEL	              0x0
#define CL_AVC_ME_COST_PRECISION_HPEL_INTEL	              0x1
#define CL_AVC_ME_COST_PRECISION_PEL_INTEL                0x2
#define CL_AVC_ME_COST_PRECISION_DPEL_INTEL	              0x3

#define CL_AVC_ME_BIDIR_WEIGHT_QUARTER_INTEL                      0x10
#define CL_AVC_ME_BIDIR_WEIGHT_THIRD_INTEL    	                  0x15
#define CL_AVC_ME_BIDIR_WEIGHT_HALF_INTEL                         0x20
#define CL_AVC_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL                    0x2B
#define CL_AVC_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL                0x30

#define CL_AVC_ME_BORDER_REACHED_LEFT_INTEL                       0x0
#define CL_AVC_ME_BORDER_REACHED_RIGHT_INTEL                      0x2
#define CL_AVC_ME_BORDER_REACHED_TOP_INTEL                        0x4
#define CL_AVC_ME_BORDER_REACHED_BOTTOM_INTEL                     0x8

#define CL_AVC_ME_SKIP_BLOCK_PARTITION_16x16_INTEL                0x0
#define CL_AVC_ME_SKIP_BLOCK_PARTITION_8x8_INTEL                  0x4000

#define CL_AVC_ME_SKIP_BLOCK_16x16_FORWARD_ENABLE_INTEL           ( 0x1 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_16x16_BACKWARD_ENABLE_INTEL          ( 0x2 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_16x16_DUAL_ENABLE_INTEL              ( 0x3 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_FORWARD_ENABLE_INTEL             ( 0x55 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_BACKWARD_ENABLE_INTEL            ( 0xAA << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_DUAL_ENABLE_INTEL                ( 0xFF << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_0_FORWARD_ENABLE_INTEL           ( 0x1 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_0_BACKWARD_ENABLE_INTEL          ( 0x2 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_1_FORWARD_ENABLE_INTEL           ( 0x1 << 26 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_1_BACKWARD_ENABLE_INTEL          ( 0x2 << 26 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_2_FORWARD_ENABLE_INTEL           ( 0x1 << 28 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_2_BACKWARD_ENABLE_INTEL          ( 0x2 << 28 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_3_FORWARD_ENABLE_INTEL           ( 0x1 << 30 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_3_BACKWARD_ENABLE_INTEL          ( 0x2 << 30 )

#define CL_AVC_ME_BLOCK_BASED_SKIP_4x4_INTEL                      0x00
#define CL_AVC_ME_BLOCK_BASED_SKIP_8x8_INTEL                      0x80

#define CL_AVC_ME_INTRA_16x16_INTEL                               0x0
#define CL_AVC_ME_INTRA_8x8_INTEL                                 0x1
#define CL_AVC_ME_INTRA_4x4_INTEL                                 0x2

#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_16x16_INTEL           0x6
#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_8x8_INTEL             0x5
#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_4x4_INTEL             0x3 

#define CL_AVC_ME_INTRA_NEIGHBOR_LEFT_MASK_ENABLE_INTEL           0x60
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_MASK_ENABLE_INTEL          0x10
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_RIGHT_MASK_ENABLE_INTEL    0x8
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_LEFT_MASK_ENABLE_INTEL     0x4

#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL              0x0
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL            0x1
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DC_INTEL                    0x2
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL    0x3
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL   0x4
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL	              0x4
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL        0x5
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL       0x6
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL         0x7
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL         0x8
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_DC_INTEL                  0x0
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL          0x1
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL            0x2
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL               0x3
          
#define CL_AVC_ME_FRAME_FORWARD_INTEL                             0x1
#define CL_AVC_ME_FRAME_BACKWARD_INTEL                            0x2
#define CL_AVC_ME_FRAME_DUAL_INTEL                                0x3

#define CL_AVC_ME_SLICE_TYPE_PRED_INTEL                           0x0
#define CL_AVC_ME_SLICE_TYPE_BPRED_INTEL                          0x1
#define CL_AVC_ME_SLICE_TYPE_INTRA_INTEL                          0x2

#define CL_AVC_ME_INTERLACED_SCAN_TOP_FIELD_INTEL                 0x0
#define CL_AVC_ME_INTERLACED_SCAN_BOTTOM_FIELD_INTEL              0x1
  
#ifdef __cplusplus
}
#endif


#endif /* __CL_INTEL_DEVICE_SIDE_AVC_MOTION_ESTIMATION_H */
