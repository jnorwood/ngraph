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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>

//#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;


using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

int load_onnx(int argc, char** argv)
{
	//TODO this needs to pass Weights as second arg
    auto function =
        onnx_import::import_onnx_model(argv[1]);

    Inputs inputs{{}};
    auto parms = function->get_parameters();

    int input_sz= parms.size();
    if (input_sz == 0){
         throw ngraph::ngraph_error("expecting input 0 is image");
    }
    // going to assume first element is the image input...   
    auto image_type = parms.at(0)->get_element_type();
    if (image_type != element::f32){
        throw ngraph::ngraph_error("expecting fp32 image type");
    }
    auto image_shape =  parms.at(0)->get_shape();
    int shape_dim = image_shape.size();
    unsigned long image_size = 1;
    for (int i = 0; i < shape_dim; i++){
        image_size *= image_shape[i] ; 
    }
    inputs[0].assign(image_size,0.5);
    //TODO load an image, from NDArray?

    auto results = function->get_results();
    int results_sz= results.size();
    if (results_sz == 0){
        throw ngraph::ngraph_error("expecting non-zero results size");
    }
    auto results_type = results.at(0)->get_element_type();
    if (results_type != element::f32){
        throw ngraph::ngraph_error("expecting fp32 results type");
    }
    auto results_shape =  results.at(0)->get_shape();
    int results_dim = results_shape.size();
    unsigned long results_size = 1;
    for (int i = 0; i < results_dim; i++){
        results_size *= results_shape[i] ; 
    }
    Outputs expected_outputs{{}};
    expected_outputs[0].assign(results_size,0.5);
    
    // BACKEND_NAME can be CPU or INTERPRETER, but INTERPRETER IS ABOUT 100X SLOWER
    Outputs outputs{execute(function, inputs, "CPU")};
	auto front = outputs.front();
	return 0;

    // TODO  would need to expect a particular category selected by some threshold
    //EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

using namespace std;

int main(int argc, char** argv)
{
    //const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    //argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc++;

    //::testing::InitGoogleTest(&argc, argv_vector.data());
    auto start = std::chrono::system_clock::now();
    int rc = load_onnx(argc,argv_vector.data());
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start);
    //NGRAPH_DEBUG_PRINT("[MAIN] Tests finished: Time: %d ms Exit code: %d", elapsed.count(), rc);

    return rc;
}


