#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define REPEAT 10000

int polling(void* addr);

int polling(void* addr) {
    printf("VA clflush %p\n", addr); 

    asm volatile (
        "clflush 0(%0)"
        :
        : "r" (addr)
    );

    return 0;
}
