#include "micro_time.h"
#include "stm32f4xx.h"

#define CPU_FREQ_HZ 84000000U

void timing_init() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t timing_get_cycles() {
    return DWT->CYCCNT;
}

float timing_cycles_to_ms(uint32_t cycles) {
    return (float)cycles / (CPU_FREQ_HZ / 1000U);
}