// #include <cstdio>
// #include <cstdarg>
// #include <cstring>

// #include "system_setup.h"
// #include "micro_time.h"
// #include "model_data.h"

// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "op_resolver.h"

// constexpr int kArenaSize = 64 * 1024;
// static uint8_t tensor_arena[kArenaSize];
// constexpr int kNumRuns = 100;

// #define PIPELINE_NAME "P_QAT_KD"

// extern "C" int _write(int fd, const char* buf, int len);

// static void uart_print(const char* s) {
//     int len = 0;
//     while (s[len]) len++;
//     _write(1, s, len);
// }

// static void uart_printf(const char* fmt, ...) {
//     char buf[128];
//     va_list args;
//     va_start(args, fmt);
//     vsnprintf(buf, sizeof(buf), fmt, args);
//     va_end(args);
//     int len = 0;
//     while (buf[len]) len++;
//     _write(1, buf, len);
// }

// static tflite::MicroMutableOpResolver<14> resolver;
// static tflite::MicroInterpreter* interpreter_ptr = nullptr;
// static uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];

// int main() {
//     system_setup();
//     uart_print("BOOT\r\n");
//     timing_init();

//     const tflite::Model* model = tflite::GetModel(g_model_data);
//     if (model == nullptr) {
//         uart_print("ERROR: model is null\r\n");
//         return 1;
//     }

//     if (model->version() != TFLITE_SCHEMA_VERSION) {
//         uart_print("ERROR: schema mismatch\r\n");
//         return 1;
//     }

//     populate_op_resolver(resolver);

//     interpreter_ptr = new (interpreter_buf) tflite::MicroInterpreter(
//         model, resolver, tensor_arena, kArenaSize
//     );

//     TfLiteStatus alloc_status = interpreter_ptr->AllocateTensors();
//     if (alloc_status != kTfLiteOk) {
//         uart_printf("ERROR: AllocateTensors failed status=%d\r\n", (int)alloc_status);
//         return 1;
//     }

//     size_t arena_used = interpreter_ptr->arena_used_bytes();

//     TfLiteTensor* input = interpreter_ptr->input(0);
//     if (input == nullptr) {
//         uart_print("ERROR: input tensor is null\r\n");
//         return 1;
//     }
//     memset(input->data.raw, 0, input->bytes);

//     // warmup
//     interpreter_ptr->Invoke();

//     // timed loop
//     uint32_t t_start = timing_get_cycles();
//     for (int i = 0; i < kNumRuns; i++) {
//         TfLiteStatus s = interpreter_ptr->Invoke();
//         if (s != kTfLiteOk) {
//             uart_printf("ERROR: Invoke failed at run %d\r\n", i);
//             return 1;
//         }
//         if ((i + 1) % 10 == 0) {
//             uart_printf("COUNT: %d\r\n", i + 1);
//         }
//     }
//     uint32_t t_end = timing_get_cycles();

//     uint32_t avg_cycles = (t_end - t_start) / kNumRuns;
//     float    avg_ms     = timing_cycles_to_ms(avg_cycles);

//     uart_printf("RESULT,%s,%lu,%.3f,%u\r\n",
//                 PIPELINE_NAME,
//                 (unsigned long)avg_cycles,
//                 avg_ms,
//                 (unsigned int)arena_used);

//     uart_print("DONE\r\n");

//     while (1) {}
// }

#include <cstdio>
#include <cstdarg>
#include <cstring>

#include "system_setup.h"
#include "micro_time.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "op_resolver.h"

constexpr int kArenaSize = 64 * 1024;
static uint8_t tensor_arena[kArenaSize];
constexpr int kNumRuns = 100;

#ifndef PIPELINE_NAME
#define PIPELINE_NAME "unknown"
#endif

extern "C" int _write(int fd, const char* buf, int len);

static void uart_print(const char* s) {
    int len = 0;
    while (s[len]) len++;
    _write(1, s, len);
}

static void uart_printf(const char* fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    int len = 0;
    while (buf[len]) len++;
    _write(1, buf, len);
}

int main() {
    system_setup();
    uart_print("BOOT\r\n");
    timing_init();
    uart_print("PRE_GETMODEL\r\n"); 

    static tflite::MicroMutableOpResolver<14> resolver;
    static uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];
    static tflite::MicroInterpreter* interpreter_ptr = nullptr;

    uart_print("STATICS_DONE\r\n");

    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        uart_print("ERROR: schema mismatch\r\n");
        return 1;
    }

    uart_print("GETMODEL_DONE\r\n");

    populate_op_resolver(resolver);

    interpreter_ptr = new (interpreter_buf) tflite::MicroInterpreter(
        model, resolver, tensor_arena, kArenaSize
    );

    if (interpreter_ptr->AllocateTensors() != kTfLiteOk) {
        uart_print("ERROR: AllocateTensors failed\r\n");
        return 1;
    }

    size_t arena_used = interpreter_ptr->arena_used_bytes();

    TfLiteTensor* input = interpreter_ptr->input(0);
    memset(input->data.raw, 0, input->bytes);

    // warmup
    interpreter_ptr->Invoke();

    // timed loop
    uint32_t t_start = timing_get_cycles();
    for (int i = 0; i < kNumRuns; i++) {
        interpreter_ptr->Invoke();
        if ((i + 1) % 10 == 0) {
            uart_printf("COUNT: %d\r\n", i + 1);
        }
    }
    uint32_t t_end = timing_get_cycles();

    uint32_t avg_cycles = (t_end - t_start) / kNumRuns;
    float    avg_ms     = timing_cycles_to_ms(avg_cycles);

    uart_printf("RESULT,%s,%lu,%.3f,%u\r\n",
                PIPELINE_NAME,
                (unsigned long)avg_cycles,
                avg_ms,
                (unsigned int)arena_used);

    uart_print("DONE\r\n");

    while (1) {}
}