#pragma once
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Registers all ops used by the timeseries TCN model.
// Conv1D is lowered to Conv2D internally by the TFLite converter.
inline void populate_op_resolver(tflite::MicroMutableOpResolver<14>& resolver) {
    resolver.AddPad();
    resolver.AddExpandDims();
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddSpaceToBatchNd();
    resolver.AddBatchToSpaceNd();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddStridedSlice();
    resolver.AddFullyConnected();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddRelu();
    resolver.AddMean();
}
