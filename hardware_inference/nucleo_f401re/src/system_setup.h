#pragma once

// Initialises system clock to 84 MHz and USART2 at 115200 baud.
// USART2 TX is on PA2, connected to ST-LINK virtual COM port.
void system_setup();
