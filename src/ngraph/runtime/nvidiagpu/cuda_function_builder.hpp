//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <string>

#include "ngraph/runtime/nvidiagpu/cuda_context_manager.hpp"
#include "ngraph/runtime/nvidiagpu/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvidiagpu
        {
            class CudaFunctionBuilder
            {
            public:
                static std::shared_ptr<CUfunction> get(const std::string& name,
                                                       const std::string& kernel,
                                                       int number_of_options,
                                                       const char** options);
            };
        }
    }
}