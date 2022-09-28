#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <linux/mman.h>

#define ARRAYSIZE 2147483648L

// #define MYPAGESIZE 1073741824UL      // 1GB
// #define NUMPAGES 2L
// #define PAGES_MAPPED 2L

#define MYPAGESIZE 2097152L     // 2MB
#define NUMPAGES 1024L
#define PAGES_MAPPED 14L

// #define MYPAGESIZE 4096L     // 4KB
// #define NUMPAGES 524288L
// #define PAGES_MAPPED 14L

// va2pa_lib.c interface
void print_pagemap_entry(unsigned long long pagemap_entry);
unsigned long long get_pagemap_entry(void *va);

// cache_latency.c interface
int cache_latency(double *address);

double *array;
double *page_pointers[NUMPAGES];
uint64_t pageframenumber[NUMPAGES];

uint64_t paddr_by_page[PAGES_MAPPED];

int main(int argc, char *argv[])
{
    size_t len;
    long arraylen;
    unsigned long pagemapentry;
    unsigned long paddr, basephysaddr;
    long j, k;

    len = NUMPAGES * MYPAGESIZE;
    printf("len: %ldMB\n", len/(1UL<<20));
    // Change later to use super page
    array = (double*) mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1 , 0);
    // array = (double*) mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1 , 0);

    if (array == (void *)(-1)) {
        perror("mmap");
        exit(1);
    }

    printf("mmap start address! %p\n", array);

    arraylen = NUMPAGES * MYPAGESIZE/sizeof(double);

    // Allocate memory
#pragma omp parallel for
    for (j=0; j<arraylen; j++) {
        array[j] = 1.0;
    }

    // Create eviction sets


    // Get physical address
    for (j=0; j<NUMPAGES; j++) {
        k = j*MYPAGESIZE/sizeof(double);
        page_pointers[j] = &array[k];
        pagemapentry = get_pagemap_entry(&array[k]);
        pageframenumber[j] = (pagemapentry & (unsigned long) 0x007FFFFFFFFFFFFF);
        // printf(" %.5ld   %.10ld  %18p  %#18lx  %#18lx  %#18lx\n",j,k,&array[k],pagemapentry,pageframenumber[j],(pageframenumber[j]<<12));
        printf("VA:  %18p     pagemap_entry:  %#18lx     PA:  %#18lx\n", &array[k], pagemapentry, (pageframenumber[j]<<12));
    }

    // Estimate L3 latency
    // cache_latency(&array[0]);

    // print page addresses (just a few of them?)
    printf("PAGE_ADDRESSES ");
    for (j=0; j<PAGES_MAPPED; j++) {
        basephysaddr = pageframenumber[j] << 12;
        paddr_by_page[j] = basephysaddr;
        printf("0x%.12lx ", paddr_by_page[j]);
    }
    printf("\n");

    // unmap memory
    int err = munmap(array, len);
    if(err != 0){
        printf("UnMapping Failed\n");
        return 1;
    }

    return 0;
}