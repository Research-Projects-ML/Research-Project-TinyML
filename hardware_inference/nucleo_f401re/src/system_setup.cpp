#include "system_setup.h"
#include "stm32f4xx_hal.h"
#include <cstdio>

// ── Clock ─────────────────────────────────────────────────────────────────────
static void clock_init() {
    RCC_OscInitTypeDef osc = {};
    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    osc.HSIState       = RCC_HSI_ON;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.PLL.PLLState   = RCC_PLL_ON;
    osc.PLL.PLLSource  = RCC_PLLSOURCE_HSI;
    osc.PLL.PLLM       = 16;
    osc.PLL.PLLN       = 336;
    osc.PLL.PLLP       = RCC_PLLP_DIV4;
    osc.PLL.PLLQ       = 7;
    HAL_RCC_OscConfig(&osc);

    RCC_ClkInitTypeDef clk = {};
    clk.ClockType      = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                         RCC_CLOCKTYPE_PCLK1  | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV2;
    clk.APB2CLKDivider = RCC_HCLK_DIV1;
    HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_2);
}

// ── USART2 ────────────────────────────────────────────────────────────────────
static UART_HandleTypeDef huart2;

static void uart_init() {
    __HAL_RCC_USART2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitTypeDef gpio = {};
    gpio.Pin       = GPIO_PIN_2;
    gpio.Mode      = GPIO_MODE_AF_PP;
    gpio.Pull      = GPIO_NOPULL;
    gpio.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &gpio);

    huart2.Instance          = USART2;
    huart2.Init.BaudRate     = 115200;
    huart2.Init.WordLength   = UART_WORDLENGTH_8B;
    huart2.Init.StopBits     = UART_STOPBITS_1;
    huart2.Init.Parity       = UART_PARITY_NONE;
    huart2.Init.Mode         = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    HAL_UART_Init(&huart2);
}

extern "C" int __io_putchar(int ch) {
    HAL_UART_Transmit(&huart2, (uint8_t*)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

extern "C" int _write(int fd, const char* buf, int len) {
    HAL_UART_Transmit(&huart2, (uint8_t*)buf, len, HAL_MAX_DELAY);
    return len;
}

extern "C" void HardFault_Handler(void) {
    const char* msg = "FAULT:HARD\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, 12, HAL_MAX_DELAY);
    while (1) {}
}

extern "C" void BusFault_Handler(void) {
    const char* msg = "FAULT:BUS\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, 11, HAL_MAX_DELAY);
    while (1) {}
}

extern "C" void UsageFault_Handler(void) {
    const char* msg = "FAULT:USAGE\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, 13, HAL_MAX_DELAY);
    while (1) {}
}

extern "C" void MemManage_Handler(void) {
    const char* msg = "FAULT:MEM\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, 11, HAL_MAX_DELAY);
    while (1) {}
}

void system_setup() {
    HAL_Init();
    clock_init();
    uart_init();
}