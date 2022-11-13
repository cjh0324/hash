#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define REPEAT 10000

int polling(void* addr);

int polling(void* addr) {
    // printf("VA clflush %p\n", addr); 
    int i;

    for (i=0; i<REPEAT; i++) {
        __asm__ __volatile__ (
            "mfence\n\t"
            "clflush 0(%0)\n\t"
            "mfence\n\t"
            :
            : "r" (addr)
        );
    }

    return 0;
}
