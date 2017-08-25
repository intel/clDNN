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

#include "kernel_selector_common.h"
#include "kernel_selector.h"
#include <type_traits>
#include <iostream>
#include <sstream>
#include <fstream>

// #define ENABLE_ENV
// #define ENABLE_ENV_PRINT

#ifdef ENABLE_ENV_PRINT
#define ENV_PRINTF(...) printf(__VA_ARGS__)
#else
#define ENV_PRINTF(...) 
#endif // ENABLE_ENV_PRINT

 
namespace KernelSelector {

#ifdef ENABLE_ENV
    std::string strip(const std::string str)
    {
        size_t start = str.find_first_not_of(' ');
        size_t end = str.find_last_not_of(' ');
        if (start == std::string::npos ||
            end == std::string::npos)
        {
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static void AddToForceMap(ForceList& force_list, bool force_or_deny, const char* env_str)
    {
        std::stringstream ss;
        ss.str(GetStringEnv(env_str));

        ENV_PRINTF("ENV: %s = %s\n", env_str, ss.str().c_str());

        std::string val;
        while (std::getline(ss, val, ','))
        {
            std::string kernel_name = strip(val);
            if (!kernel_name.empty())
            {
                force_list[kernel_name] = force_or_deny;
            }
        }
    }
#endif

    KernelSelctorBase::KernelSelctorBase()
    {
#ifdef ENABLE_ENV
        AddToForceMap(forceKernels, true, "CL_DNN_FORCE_KERNELS");
        AddToForceMap(forceKernels, false, "CL_DNN_DENY_KERNELS");
#endif
    }

    KernelsData KernelSelctorBase::GetNaiveBestKernel(const Params& params, const OptionalParams& options, KernelType kType) const
    {
        KernelsData kernelsData;
        std::string kernelName;

        if (params.GetType() == kType &&
            options.GetType() == kType)
        {
            const ParamsKey requireKey = params.GetParamsKey().Merge(options.GetSupportedKey());
            for (const auto& implementation : implementations)
            {
                const ParamsKey implKey = implementation->GetSupportedKey();
                if (implKey.Support(requireKey))
                {
                    try
                    {
                        KernelsData kds = implementation->GetKernelsData(params, options);

                        if (kds.size() && kds[0].kernels.size())
                        {
#ifdef ENABLE_ENV
                            const auto& it = forceKernels.find(implementation->GetName());
                            if (it != forceKernels.end())
                            {
                                if (it->second == true)
                                {
                                    ENV_PRINTF("Force: %s\n", it->first.c_str());
                                    return kds;
                                }
                                else
                                {
                                    ENV_PRINTF("Deny: %s\n", it->first.c_str());
                                }
                            }
                            else
#endif
                            {
                                if (kernelsData.size() == 0 ||
                                    kds[0].estimatedTime < kernelsData[0].estimatedTime)
                                {
                                    kernelsData = kds;
                                    kernelName = implementation->GetName();
                                }
                            }
                        }
                    }
                    catch (std::runtime_error&)
                    {
                        // we have to handle it in order to avoid exception in KernelSelector as much we can
                    }
                }
            }
        }

        // TODO: find a better place to located this assignment 
        if (kernelsData.size())
        {
            //printf("%s\n", kernelName.c_str());
            kernelsData[0].kernelName = kernelName;
            kernelsData[0].kernels[0].layerID = params.layerID;
        }

        return kernelsData;
    }

    std::map<std::string, std::tuple<std::string, int>> KernelSelctorBase::LoadTunedKernels(const std::string& cacheFilePath) const
    {
        std::map<std::string, std::tuple<std::string, int>> cachedKernels;
        std::ifstream cachedKernelsFile(cacheFilePath);
        std::string hash;
        std::string cachedkernelName;
        int autoTuneIndex;
        if (cachedKernelsFile)
        {
            while (cachedKernelsFile.good())
            {
                cachedKernelsFile >> hash;
                if (!cachedKernelsFile.good())
                {
                    break;
                }
                cachedKernelsFile >> cachedkernelName;
                if (!cachedKernelsFile.good())
                {
                    break;
                }
                cachedKernelsFile >> autoTuneIndex;
                cachedKernels[hash] = std::make_tuple(cachedkernelName, autoTuneIndex);
            }
        }
        cachedKernelsFile.close();
        return cachedKernels;
    }

    void KernelSelctorBase::StoreTunedKernel(const std::string hash, const std::string name, const int autoTuneIndex, const std::string& cacheFilePath) const
    {
        std::ofstream cachedKernelsFile(cacheFilePath, std::ofstream::out | std::ofstream::app);
        cachedKernelsFile << hash << " ";
        cachedKernelsFile << name << " ";
        cachedKernelsFile << autoTuneIndex << "\n";
        cachedKernelsFile.close();
    }

    KernelsData KernelSelctorBase::GetAutoTuneBestKernel(const Params& params, const OptionalParams& options, KernelType kType) const
    {
        assert(options.tuningParams.mode != TuningMode::TUNING_DISABLED);

        KernelsData kernelsData;
        std::string kernelName;
        std::string cacheFile;

        if (params.GetType() == kType &&
            options.GetType() == kType)
        {
            auto cachedKernels = LoadTunedKernels(options.tuningParams.cacheFilePath);
            std::string hash = std::to_string(std::hash<std::string>{}(params.to_string()));

            bool hashFoundInCache = cachedKernels.find(hash) != cachedKernels.end();

            if (hashFoundInCache)
            {
                std::tuple<std::string, int> cachedKernel = cachedKernels[hash];
                std::string cachedkernelName = std::get<0>(cachedKernel);
                int autoTuneIndex = std::get<1>(cachedKernel);

                for (const auto& implementation : implementations)
                {
                    if (implementation->GetName().compare(cachedkernelName) == 0)
                    {            
                        KernelsData kds = implementation->GetTunedKernelsDataByIndex(params, options, autoTuneIndex);
                        if (kds.size() && kds[0].kernels.size())
                        {
                            kernelsData = kds;
                            kernelsData[0].kernelName = cachedkernelName;
                            kernelsData[0].kernels[0].layerID = params.layerID;
                        }
                        break;
                    }
                }

                if (!kernelsData.empty())
                {
                    return kernelsData;
                }
            }

            if( hashFoundInCache || // Cache is not valid - hash exists in cache but kernelsData was empty.
                (options.tuningParams.mode == TuningMode::TUNING_USE_CACHE) || // Cache only mode - on-line tuning is not allowed.
                !options.tuningParams.runner ) // Runner is invalid - can't run on-line tuning
            {
                // Fall back to the default path.
                return GetNaiveBestKernel(params, options, kType);
            }    

            // Start on-line tuning
            assert(options.tuningParams.runner);

            const ParamsKey requireKey = params.GetParamsKey().Merge(options.GetSupportedKey());
            for (const auto& implementation : implementations)
            {
                const ParamsKey implKey = implementation->GetSupportedKey();
                if (implKey.Support(requireKey))
                {
                    try
                    {
                        KernelsData kds = implementation->GetKernelsDataForAutoTune(params, options);
                        std::vector<uint64_t> runTimes = options.tuningParams.runner->run_kernels(kds);

                        for (size_t i = 0; i < kds.size(); i++)
                        {
                            kds[i].runTime = runTimes[i];
                            if (kernelsData.size() == 0 || kds[i].runTime < kernelsData[0].runTime)
                            {
                                kernelsData = { kds[i] };
                                kernelName = implementation->GetName();
                            }
                        }
                    }
                    catch (std::runtime_error&)
                    {
                        // we have to handle it in order to avoid exception in KernelSelector as much we can
                    }
                }
            }

            if (kernelsData.size())
            {
                kernelsData[0].kernelName = kernelName;
                kernelsData[0].kernels[0].layerID = params.layerID;
                StoreTunedKernel(hash, kernelName, kernelsData[0].autoTuneIndex, options.tuningParams.cacheFilePath);
            }
        } 

        return kernelsData;
    }
}