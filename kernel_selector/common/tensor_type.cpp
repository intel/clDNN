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

#include <cstddef>
#include "tensor_type.h"
#include "common_tools.h"

namespace KernelSelector
{
    namespace Tensor
    {
        NDims DataTensor::GetSimpleDims(const std::vector<size_t>& d, DataLayout l)
        {
            std::vector<size_t> newDims = d;

            // TOOD: it's not the right pitches. it's here in order to calculate physical size
            switch (l)
            {
            case bs_f_bsv8__af8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 8);
                break;
            case bs_f_bsv16__af8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 16);
                break;
            default:
                break;
            }

            NDims ret(newDims.size());
            size_t pitch = 1;

            for (size_t i = 0; i < newDims.size(); i++)
            {
                Pad p = { 0, newDims[i] - d[i] };
                ret[i] = { d[i], pitch, p };
                pitch *= newDims[i];
            }

            return ret;
        }

        DataTensor DataTensor::TransformIgnorePadding(DataLayout l) const
        {
            const uint32_t src_channels = ChannelsCount(layout);
            const uint32_t dst_channels = ChannelsCount(l);

            const size_t src_x = X().v;
            const size_t src_y = Y().v;

            std::vector<size_t> vec(dst_channels);
            if (src_channels == 2 && dst_channels == 2)
            {
                vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else if (src_channels == 4 && dst_channels == 4)
            {
                vec[Channelndex(l, DataChannelName::X)] = X().v;
                vec[Channelndex(l, DataChannelName::Y)] = Y().v;
                vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else if (src_channels == 2 && dst_channels == 4)
            {
                const size_t dst_ifm = Feature().v / (src_x*src_y);
                const size_t dst_xy = Feature().v % (src_x*src_y);
                const size_t dst_y = dst_xy / src_x;
                const size_t dst_x = dst_xy % src_x;
                vec[Channelndex(l, DataChannelName::X)] = dst_x;
                vec[Channelndex(l, DataChannelName::Y)] = dst_y;
                vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else if (src_channels == 4 && dst_channels == 2)
            {
                const size_t dst_ifm = Feature().v * src_x * src_y;
                vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
                vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
            }
            else
            {
                // TODO: implement ROI
                assert(0);
            }

            return{ vec, dtype, l };
        }

        DataTensor DataTensor::FlattenFeatureAndSpatials() const
        {
            DataLayout l;

            const auto x = X();
            const auto y = Y();
            const auto f = Feature();
            const auto b = Batch();

            DataLayout targetLayout = Tensor::bf;
            switch (layout)
            {
            case Tensor::bf:
            case Tensor::fb:
                return *this;
            case Tensor::fyxb:
                targetLayout = Tensor::fb;
            case Tensor::bfyx:
                if (f.pitch == y.v*x.v*x.pitch)                                         // no padding in X/Y axis
                {
                    l = targetLayout;
                    break;
                }
                throw std::runtime_error("Unsupported - cannot flatten with padding");
            case Tensor::yxfb:
                targetLayout = Tensor::fb;
            case Tensor::byxf:
                if ((x.pitch == f.pitch && y.pitch == x.v*x.pitch) ||                   // YX - no Features (val/pitch)
                    (y.v == 1 && x.v == 1 && x.pitch == f.pitch && y.pitch == f.pitch)) // Feature only
                {
                    l = targetLayout;
                    break;
                }
                throw std::runtime_error("Unsupported - cannot flatten yxf to f if f/yx != 1");
            default:
                throw std::runtime_error("Unsupported - unsupported layout");
                break;
            }

            DataTensor res = TransformIgnorePadding(l);

            if (l == DataLayout::bf)
            {
                res.dims[Channelndex(l, DataChannelName::BATCH)].pitch = b.pitch;
                res.dims[Channelndex(l, DataChannelName::BATCH)].pad   = b.pad;
            }
            else
            {
                res.dims[Channelndex(l, DataChannelName::FEATURE)].pitch = dims[Channelndex(l, DataChannelName::BATCH) + 1].pitch;
                res.dims[Channelndex(l, DataChannelName::FEATURE)].pad   = dims[Channelndex(l, DataChannelName::BATCH) + 1].pad;
            }

            return res;
        }

        NDims WeightsTensor::GetSimpleDims(const std::vector<size_t>& d, WeightsLayout l)
        {
            std::vector<size_t> newDims = d;

            // TOOD: it's not the right pitches. it's here in order to calculate physical size
            switch (l)
            {
            case os_iyx_osv16:
                assert(newDims.size() == 4);
                newDims[3] = RoundUp(newDims[3], 16);
                break;
            case os_i_osv8__ai8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 8);
                break;
            case os_i_osv16__ai8:
                assert(newDims.size() == 2);
                newDims[0] = RoundUp(newDims[0], 8);
                newDims[1] = RoundUp(newDims[1], 16);
                break;
            case os_i_osv16:
                assert(newDims.size() == 2);
                newDims[1] = RoundUp(newDims[1], 16);
                break;
            case i_yxs_os_yxsv2_osv16:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 16);
                break;
            case iy_xs_os_xsv2_osv16__ao32:
            case iy_xs_os_xsv2_osv8__ao32:
                assert(newDims.size() == 4);
                newDims[0] = RoundUp(newDims[0], 32);
                break;
            default:
                break;
            }

            NDims ret(newDims.size());
            size_t pitch = 1;

            for (size_t i = 0; i < newDims.size(); i++)
            {
                Pad p = { 0, newDims[i] - d[i] };
                ret[i] = { d[i], pitch, p };
                pitch *= newDims[i];
            }

            if (l == i_yxs_os_yxsv2_osv16)
            {
                ret[3].pitch = RoundUp(ret[1].v * ret[2].v, 2) * ret[1].pitch;
                ret[2].pad.after = newDims[2] - ret[2].v;
            }
            else if (l == iy_xs_os_xsv2_osv16__ao32 ||
                     l == iy_xs_os_xsv2_osv8__ao32)
            {
                ret[2].pitch     = RoundUp(ret[1].v, 2) * ret[1].pitch;
                ret[1].pad.after = newDims[1] - ret[1].v;
                
                ret[3].pitch     = ret[2].v * ret[2].pitch;
                ret[2].pad.after = newDims[2] - ret[2].v;
            }

            return ret;
        }

        WeightsTensor WeightsTensor::TransformIgnorePadding(WeightsLayout l, WeightsType t) const
        {
            const uint32_t src_channels = ChannelsCount(layout);
            const uint32_t dst_channels = ChannelsCount(l);

            const size_t src_x = X().v;
            const size_t src_y = Y().v;

            std::vector<size_t> vec(dst_channels);
            if (src_channels == 2 && dst_channels == 2)
            {
                vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else if (src_channels == 4 && dst_channels == 4)
            {
                vec[Channelndex(l, WeightsChannelName::X)] = X().v;
                vec[Channelndex(l, WeightsChannelName::Y)] = Y().v;
                vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else if (src_channels == 2 && dst_channels == 4)
            {
                const size_t dst_ifm = IFM().v / (src_x*src_y);
                const size_t dst_xy = IFM().v % (src_x*src_y);
                const size_t dst_y = dst_xy / src_x;
                const size_t dst_x = dst_xy % src_x;
                vec[Channelndex(l, WeightsChannelName::X)] = dst_x;
                vec[Channelndex(l, WeightsChannelName::Y)] = dst_y;
                vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else if (src_channels == 4 && dst_channels == 2)
            {
                const size_t dst_ifm = IFM().v * src_x * src_y;
                vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
                vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
            }
            else
            {
                assert(0);
            }

            return{ vec, t, l };
        }
    }
}