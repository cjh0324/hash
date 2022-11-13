#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <linux/mman.h>

#define ARRAYSIZE 2147483648L
#define CACHESIZE 64L
#define NUMCHAS   18

// #define MYPAGESIZE 1073741824UL      // 1GB
// #define NUMPAGES 2L
// #define PAGES_MAPPED 2L

#define MYPAGESIZE 2097152L     // 2MB
#define NUMPAGES 2L
#define PAGES_MAPPED 14L

// #define MYPAGESIZE 4096L     // 4KB
// #define NUMPAGES 524288L
// #define PAGES_MAPPED 14L

// va2pa_lib.c interface
void print_pagemap_entry(unsigned long long pagemap_entry);
unsigned long long get_pagemap_entry(void *va);

// cache_latency.c interface
int cache_latency(double *address);

// polling.c interface
int polling(void* address);

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
    long i, j, k;
    char filename[100];
    char buffer[120];
    int pkg, tile;
    int nr_cpus;
    uint64_t msr_val, msr_num;
    uint64_t counter[2];
    int msr_fd;
    int mapping_fd;
    int proc_in_pkg[2];

    len = NUMPAGES * MYPAGESIZE;
    printf("len: %ldMB\n", len/(1UL<<20));
    // Change later to use super page
    // array = (double*) mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB, -1 , 0);
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
        // printf("VA:  %18p     pagemap_entry:  %#18lx     PA:  %#18lx\n", &array[k], pagemapentry, (pageframenumber[j]<<12));
    }

    // Estimate L3 latency
    // cache_latency(&array[0]);
   
    sprintf(filename, "/dev/cpu/0/msr");
    msr_fd = open(filename, O_RDWR);
    if (msr_fd == -1) {
        fprintf(stderr, "ERROR %s when trying to open %s\n", strerror(errno), filename);
        exit(-1);
    }
    sprintf(filename, "txt/mapping");
    mapping_fd = open(filename, O_RDWR | O_APPEND | O_CREAT);
    if (mapping_fd == -1) {
        fprintf(stderr, "ERROR %s when trying to open %s\n", strerror(errno), filename);
        exit(-1);
    }

    pread(msr_fd, &msr_val, sizeof(msr_val), 0x186L);
    printf("Core PerfEvtSel0: %ld\n", msr_val);
    // pwrite test
    msr_val = 0x0L;
    if (!pwrite(msr_fd, &msr_val, sizeof(msr_val), 0xE18)) {
        perror("pwrite test failed!");
    }
    // Configure LLC_LOOKUP event in all CHAs
    for (tile=0; tile<NUMCHAS; tile++) {
        msr_num = 0xe00 + 0x10 * tile;
        msr_val = 0x00400000;
        pwrite(msr_fd, &msr_val, sizeof(msr_val), msr_num);
        msr_num = 0xe00 + 0x10 * tile + 1;
        msr_val = 0x00401f34;
        pwrite(msr_fd, &msr_val, sizeof(msr_val), msr_num);
        msr_num = 0xe00 + 0x10 * tile + 5;
        msr_val = 0x01e20000;
        pwrite(msr_fd, &msr_val, sizeof(msr_val), msr_num);
    }
    
    // Polling to get cache slice
    printf("For Test! First page addresses\n");
    // for (i=0; i<NUMPAGES; i++) {
    for (i=0; i<1; i++) {
        k = i * MYPAGESIZE/sizeof(double) - CACHESIZE/sizeof(double);
        // for (j=0; j<MYPAGESIZE/CACHESIZE; j++) {
        for (j=0; j<10; j++) {
            k += CACHESIZE/sizeof(double);
            // Initialize counted values
            for (tile=0; tile<NUMCHAS; tile++) {
                msr_num = 0xe00 + 0x10 * tile + 0x8;
                msr_val = 0x0L;
                pwrite(msr_fd, &msr_val, sizeof(msr_val), msr_num);
            }
            // POLLING
            polling(&array[k]);
            // Read all counter registers
            counter[0] = 0;
            for (tile=0; tile<NUMCHAS; tile++) {
                msr_num = 0xe00 + 0x10 * tile + 0x8;
                pread(msr_fd, &msr_val, sizeof(msr_val), msr_num);
                if (msr_val > counter[0]) {
                  counter[1] = tile+1;
                  counter[0] = msr_val;
                }
                printf("Slice %d LLC_LOOKUP count: %ld\n", tile, msr_val);
            }
            // 2MB
            paddr = pageframenumber[i]<<12 | (uintptr_t)&array[k] & 0x1FFFFF;
            // 1GB
            // paddr = pageframenumber[i]<<12 | (uintptr_t)&array[k] & 0x3FFFFFFF;
            sprintf(buffer, "%lx\t%ld\n",paddr,counter[1]);
            if (!write(mapping_fd, buffer, strlen(buffer))) {
                perror("Cannot write to file.");
                exit(1);
            }
        }
    }

    // Initialize event selector
    msr_val = 0x0L;
    for (tile=0; tile<NUMCHAS; tile++) {
        msr_num = 0xe00 + 0x10 * tile;
        pwrite(msr_fd, &msr_val, sizeof(msr_val), msr_num);
    }
    

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
