#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define REPEAT 10000

int polling(void* addr);

int polling(void* addr) {
    // printf("VA clflush %p\n", addr); 

    for (int i=0; i<REPEAT; i++) {
        __asm__ __volatile__ (
            "clflush 0(%0)"
            :
            : "r" (addr)
        );
    }

    return 0;
}
