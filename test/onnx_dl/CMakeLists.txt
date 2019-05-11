cmake_minimum_required(VERSION 3.13)
include(FetchContent)
include(ExternalProject)
set(CMAKE_BUILD_TYPE Debug)

option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})
    enable_testing()
endif()
set(FETCHCONTENT_QUIET OFF)

# Source directories
set(ONNX_MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/models_onnx)
set(ONNX_EXEC_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)


	
FetchContent_Declare(
  bvlc_alexnet
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz
  SOURCE_DIR  ${ONNX_MODELS_DIR}/bvlc_alexnet
)

FetchContent_GetProperties(bvlc_alexnet)
if(NOT bvlc_alexnet_POPULATED)
  FetchContent_Populate(bvlc_alexnet)
endif()


FetchContent_Declare(
  bvlc_googlenet
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz
  SOURCE_DIR  ${ONNX_MODELS_DIR}/bvlc_googlenet
)

FetchContent_GetProperties(bvlc_googlenet)
if(NOT bvlc_googlenet_POPULATED)
  FetchContent_Populate(bvlc_googlenet)
endif()

FetchContent_Declare(
  bvlc_reference_caffenet
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/bvlc_reference_caffenet
)

FetchContent_GetProperties(bvlc_reference_caffenet)
if(NOT bvlc_reference_caffenet_POPULATED)
  FetchContent_Populate(bvlc_reference_caffenet)
endif()

FetchContent_Declare(
  bvlc_reference_rcnn_ilsvrc13
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_rcnn_ilsvrc13.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/bvlc_reference_rcnn_ilsvrc13
)

FetchContent_GetProperties(bvlc_reference_rcnn_ilsvrc13)
if(NOT bvlc_reference_rcnn_ilsvrc13_POPULATED)
  FetchContent_Populate(bvlc_reference_rcnn_ilsvrc13)
endif()


FetchContent_Declare(
  densenet121
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/densenet121.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/densenet121
)

FetchContent_GetProperties(densenet121)
if(NOT densenet121_POPULATED)
  FetchContent_Populate(densenet121)
endif()

FetchContent_Declare(
  inception_v1
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/inception_v1
)

FetchContent_GetProperties(inception_v1)
if(NOT inception_v1_POPULATED)
  FetchContent_Populate(inception_v1)
endif()

FetchContent_Declare(
  inception_v2
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/inception_v2
)

FetchContent_GetProperties(inception_v2)
if(NOT inception_v2_POPULATED)
  FetchContent_Populate(inception_v2)
endif()


FetchContent_Declare(
  resnet50
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet50
)

FetchContent_GetProperties(resnet50)
if(NOT resnet50_POPULATED)
  FetchContent_Populate(resnet50)
endif()

FetchContent_Declare(
  resnet50_opset_8
  URL https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet50_opset_8
)

FetchContent_GetProperties(resnet50_opset_8)
if(NOT resnet50_opset_8_POPULATED)
  FetchContent_Populate(resnet50_opset_8)
endif()


FetchContent_Declare(
  shufflenet
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/shufflenet
)

FetchContent_GetProperties(shufflenet)
if(NOT shufflenet_POPULATED)
  FetchContent_Populate(shufflenet)
endif()


FetchContent_Declare(
  vgg19
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/vgg19
)

FetchContent_GetProperties(vgg19)
if(NOT vgg19_POPULATED)
  FetchContent_Populate(vgg19)
endif()


FetchContent_Declare(
  zfnet512
  URL https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/zfnet512
)

FetchContent_GetProperties(zfnet512)
if(NOT zfnet512_POPULATED)
  FetchContent_Populate(zfnet512)
endif()



FetchContent_Declare(
  mobilenetv2-1.0
  URL https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/mobilenetv2-1.0
)

FetchContent_GetProperties(mobilenetv2-1.0)
if(NOT mobilenetv2-1.0_POPULATED)
  FetchContent_Populate(mobilenetv2-1.0)
endif()




FetchContent_Declare(
  resnet18v1
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet18v1
)

FetchContent_GetProperties(resnet18v1)
if(NOT resnet18v1_POPULATED)
  FetchContent_Populate(resnet18v1)
endif()



FetchContent_Declare(
  resnet34v1
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet34v1
)

FetchContent_GetProperties(resnet34v1)
if(NOT resnet34v1_POPULATED)
  FetchContent_Populate(resnet34v1)
endif()



FetchContent_Declare(
  resnet50v1
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet50v1
)

FetchContent_GetProperties(resnet50v1)
if(NOT resnet50v1_POPULATED)
  FetchContent_Populate(resnet50v1)
endif()


FetchContent_Declare(
  resnet101v1
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet101v1
)

FetchContent_GetProperties(resnet101v1)
if(NOT resnet101v1_POPULATED)
  FetchContent_Populate(resnet101v1)
endif()



FetchContent_Declare(
  resnet152v1
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet152v1
)

FetchContent_GetProperties(resnet152v1)
if(NOT resnet152v1_POPULATED)
  FetchContent_Populate(resnet152v1)
endif()

FetchContent_Declare(
  resnet18v2
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet18v2
)

FetchContent_GetProperties(resnet18v2)
if(NOT resnet18v2_POPULATED)
  FetchContent_Populate(resnet18v2)
endif()

FetchContent_Declare(
  resnet34v2
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet34v2
)

FetchContent_GetProperties(resnet34v2)
if(NOT resnet34v2_POPULATED)
  FetchContent_Populate(resnet34v2)
endif()

FetchContent_Declare(
  resnet50v2
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet50v2
)

FetchContent_GetProperties(resnet50v2)
if(NOT resnet50v2_POPULATED)
  FetchContent_Populate(resnet50v2)
endif()

FetchContent_Declare(
  resnet101v2
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet101v2
)

FetchContent_GetProperties(resnet101v2)
if(NOT resnet101v2_POPULATED)
  FetchContent_Populate(resnet101v2)
endif()

FetchContent_Declare(
  resnet152v2
  URL https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/resnet152v2
)

FetchContent_GetProperties(resnet152v2)
if(NOT resnet152v2_POPULATED)
  FetchContent_Populate(resnet152v2)
endif()


FetchContent_Declare(
  squeezenet1.1
  URL https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/squeezenet1.1
)

FetchContent_GetProperties(squeezenet1.1)
if(NOT squeezenet1.1_POPULATED)
  FetchContent_Populate(squeezenet1.1)
endif()

FetchContent_Declare(
  vgg16
  URL https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/vgg16
)

FetchContent_GetProperties(vgg16)
if(NOT vgg16_POPULATED)
  FetchContent_Populate(vgg16)
endif()

FetchContent_Declare(
  vgg16-bn
  URL https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/vgg16-bn
)

FetchContent_GetProperties(vgg16-bn)
if(NOT vgg16-bn_POPULATED)
  FetchContent_Populate(vgg16-bn)
endif()


FetchContent_Declare(
  vgg19b
  URL https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/vgg19b
)

FetchContent_GetProperties(vgg19b)
if(NOT vgg19b_POPULATED)
  FetchContent_Populate(vgg19b)
endif()

FetchContent_Declare(
  vgg19-bn
  URL https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.tar.gz
  SOURCE_DIR ${ONNX_MODELS_DIR}/vgg19-bn
)

FetchContent_GetProperties(vgg19-bn)
if(NOT vgg19-bn_POPULATED)
  FetchContent_Populate(vgg19-bn)
endif()


add_test (NAME bvlc_alexnet
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/bvlc_alexnet/model.onnx bvlc_alexnet_out
)

add_test (NAME bvlc_googlenet
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/bvlc_googlenet/model.onnx bvlc_googlenet_out
)
add_test (NAME bvlc_reference_caffenet
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/bvlc_reference_caffenet/model.onnx bvlc_reference_caffenet_out
)
add_test (NAME bvlc_reference_rcnn_ilsvrc13
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/bvlc_reference_rcnn_ilsvrc13/model.onnx bvlc_reference_rcnn_ilsvrc13_out
)
add_test (NAME densenet121
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/densenet121/model.onnx densenet121_out
)

add_test (NAME inception_v1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/inception_v1/model.onnx inception_v1_out
)
add_test (NAME inception_v2
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/inception_v2/model.onnx inception_v2_out
)
add_test (NAME resnet50
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet50/model.onnx resnet50_out
)

add_test (NAME resnet50_opset_8
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet50_opset_8/model.onnx resnet50_opset_8_out
)

add_test (NAME shufflenet
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/shufflenet/model.onnx shufflenet_out
)
add_test (NAME vgg19
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/vgg19/model.onnx vgg19_out
)
add_test (NAME zfnet512
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/zfnet512/model.onnx zfnet512_out
)

add_test (NAME mobilenetv2-1.0
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/mobilenetv2-1.0/mobilenetv2-1.0.onnx mobilenetv2-1.0_out
)

add_test (NAME resnet18v1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet18v1/resnet18v1/resnet18v1.onnx resnet18v1_out
)

add_test (NAME resnet34v1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet34v1/resnet34v1/resnet34v1.onnx resnet34v1_out
)


add_test (NAME resnet50v1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet50v1/resnet50v1/resnet50v1.onnx resnet50v1_out
)

add_test (NAME resnet101v1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet101v1/resnet101v1/resnet101v1.onnx resnet101v1_out
)
add_test (NAME resnet152v1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet152v1/resnet152v1/resnet152v1.onnx resnet152v1_out
)

add_test (NAME resnet18v2
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet18v2/resnet18v2.onnx resnet18v2_out
)

add_test (NAME resnet34v2
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet34v2/resnet34v2.onnx resnet34v2_out
)


add_test (NAME resnet50v2
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet50v2/resnet50v2.onnx resnet50v2_out
)


add_test (NAME resnet101v2
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet101v2/resnet101v2.onnx resnet101v2_out
)

add_test (NAME resnet152v2
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/resnet152v2/resnet152v2.onnx resnet152v2_out
)


add_test (NAME squeezenet1.1
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/squeezenet1.1/squeezenet1.1.onnx squeezenet1.1_out
)
add_test (NAME vgg16
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/vgg16/vgg16/vgg16.onnx vgg16_out
)
add_test (NAME vgg16-bn
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/vgg16-bn/vgg16-bn/vgg16-bn.onnx vgg16-bn_out
)

add_test (NAME vgg19b
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/vgg19b/vgg19/vgg19.onnx vgg19b_out
)
add_test (NAME vgg19-bn
	COMMAND ${ONNX_EXEC_DIR}/onnx_dl_test  ${ONNX_MODELS_DIR}/vgg19-bn/vgg19-bn//vgg19-bn.onnx vgg19-bn_out
)
