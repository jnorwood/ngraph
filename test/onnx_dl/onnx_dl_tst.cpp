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



FILE* openReadFileGetSize(const char* fname, unsigned long* sizep) {
  FILE* fp;
  if ((fp = fopen(fname, "rb")) == NULL) {
    printf("### fatal error: cannot open '%s'\n", fname);
    throw ngraph::ngraph_error("### fatal error: cannot open input file ");
  }
  if (sizep != NULL) {
    fseek(fp, 0, SEEK_END);
    *sizep = ftell(fp);
    fseek(fp, 0, SEEK_SET);
  }
  return fp;
}


void readNPYFileData(const char* fname, void* buffer,
                              unsigned long size) {
  unsigned long len;
  FILE* fp = openReadFileGetSize(fname, &len);
  // Data is at the end of file, so seek backward from end of file
  fseek(fp, len - size, SEEK_SET);
  if (fread(buffer, 1, size, fp) != size) {
    throw ngraph::ngraph_error("### fatal error: input data read error ");
  }

  fclose(fp);
}


using namespace ngraph;


using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

int load_onnx(int argc, char** argv)
{
	//TODO this needs to pass Weights as second arg
    if (argc < 3){
          throw ngraph::ngraph_error("expecting arv[1]= onnx model path, argv[2]= test image");
    }
    auto function =
        onnx_import::import_onnx_model(argv[1]);
    std::vector<uint8_t> raw{};

    Inputs inputs{{}};
    auto parms = function->get_parameters();

    int input_sz= parms.size();
    if (input_sz == 0){
         throw ngraph::ngraph_error("expecting input 0 is input tensor for image");
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

    if (image_shape[3] != 224){
      printf("tst:%s, image_dim=%ld\n", argv[1], image_shape[3]);
      throw ngraph::ngraph_error("expecting 224x224 pixel image");
    }

    // for debug, loading a previously used cat image raw RGB uint8 in NHW format
    // then converting the loaded uint8 data to float 1.0 ... -1.0 range
    raw.resize(image_size);
    readNPYFileData(argv[2], raw.data(), image_size);
    inputs[0].resize(image_size);

    uint8_t * up = raw.data();
    float * fp = inputs[0].data();
    for (int i=0; i<image_size; i++){
        fp[i] = (up[i]-128.0)/128.0; // convert to +-1.0 range
    }

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
    int maxi = 0;
    float maxv = front[0];
    for (int i=1; i<results_size; i++){
        float v = front[i];
        if (v > maxv){
            maxv = v;
            maxi = i;
        }
    }
    printf("tst:%s, maxi=%d, maxv=%f\n", argv[1], maxi, maxv);
 
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


