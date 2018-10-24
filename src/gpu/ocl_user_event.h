/*
// Copyright (c) 2018 Intel Corporation
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


#pragma once

#include "ocl_base_event.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable: 4250) //Visual Studio warns us about inheritance via dominance but it's done intentionally so turn it off
#endif

namespace cldnn { namespace gpu {

struct user_event : public base_event, public cldnn::user_event
{
    user_event(std::shared_ptr<gpu_toolkit> ctx, bool auto_set = false) : base_event(ctx, cl::UserEvent(ctx->context())), cldnn::user_event(auto_set)
    {
        if (auto_set)
            user_event::set_impl();
    }

    void set_impl() override;

    bool get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) override;

protected:
    cldnn::instrumentation::timer<> _timer;
    std::unique_ptr<cldnn::instrumentation::profiling_period_basic> _duration;
};

#ifdef _WIN32
#pragma warning(pop)
#endif

} }