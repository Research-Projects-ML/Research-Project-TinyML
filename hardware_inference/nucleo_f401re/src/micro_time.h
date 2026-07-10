#pragma once
#include <stdint.h>

void     timing_init();
uint32_t timing_get_cycles();
float    timing_cycles_to_ms(uint32_t cycles);