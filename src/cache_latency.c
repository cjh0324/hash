#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sched.h>

int cache_latency(double* address);

int i386_cpuid_caches (size_t * data_caches) {
    int i;
    int num_data_caches = 0;
    for (i = 0; i < 32; i++) {

        // Variables to hold the contents of the 4 i386 legacy registers
        uint32_t eax, ebx, ecx, edx; 

        eax = 4; // get cache info
        ecx = i; // cache id

        asm (
            "cpuid" // call i386 cpuid instruction
            : "+a" (eax) // contains the cpuid command code, 4 for cache query
            , "=b" (ebx)
            , "+c" (ecx) // contains the cache id
            , "=d" (edx)
        ); // generates output in 4 registers eax, ebx, ecx and edx 

        // taken from http://download.intel.com/products/processor/manual/325462.pdf Vol. 2A 3-149
        int cache_type = eax & 0x1F; 

        if (cache_type == 0) // end of valid cache identifiers
            break;

        char * cache_type_string;
        switch (cache_type) {
            case 1: cache_type_string = "Data Cache"; break;
            case 2: cache_type_string = "Instruction Cache"; break;
            case 3: cache_type_string = "Unified Cache"; break;
            default: cache_type_string = "Unknown Type Cache"; break;
        }

        int cache_level = (eax >>= 5) & 0x7;

        int cache_is_self_initializing = (eax >>= 3) & 0x1; // does not need SW initialization
        int cache_is_fully_associative = (eax >>= 1) & 0x1;


        // taken from http://download.intel.com/products/processor/manual/325462.pdf 3-166 Vol. 2A
        // ebx contains 3 integers of 10, 10 and 12 bits respectively
        unsigned int cache_sets = ecx + 1;
        unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
        unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
        unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;

        // Total cache size is the product
        size_t cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

        if (cache_type == 1 || cache_type == 3) {
            data_caches[num_data_caches++] = cache_total_size;
        }

    }

    return num_data_caches;
}

int test_cache(size_t attempts, size_t lower_cache_size, double* address, int * latencies, size_t max_latency) {
    // printf("lower cache size %ld\n", lower_cache_size);
    // printf("sysconf pagesize? %ld\n", sysconf(_SC_PAGESIZE));
    // printf("Is address correctly inserted? %p\n", address);

    // int fd = open("/dev/urandom", O_RDONLY);
    // if (fd < 0) {
    //     perror("open");
    //     abort();
    // }
    // char * random_data = mmap(
    //       NULL
    //     , lower_cache_size
    //     , PROT_READ | PROT_WRITE
    //     , MAP_PRIVATE | MAP_ANON // | MAP_POPULATE
    //     , -1
    //     , 0
    //     ); // get some random data
    // if (random_data == MAP_FAILED) {
    //     perror("mmap");
    //     abort();
    // }

    // size_t i;
    // for (i = 0; i < lower_cache_size; i += sysconf(_SC_PAGESIZE)) {
    //     random_data[i] = 1;
    // }


    // int64_t random_offset = 0;
    while (attempts--) {
        // use processor clock timer for exact measurement
        // random_offset += rand();
        // random_offset %= lower_cache_size;
        int32_t cycles_used, edx, temp1, temp2;
        asm (
            "mfence\n\t"        // memory fence
            "rdtsc\n\t"         // get cpu cycle count
            "mov %%edx, %2\n\t"
            "mov %%eax, %3\n\t"
            "mfence\n\t"        // memory fence
            "mov %4, %%al\n\t"  // load data
            "mfence\n\t"
            "rdtsc\n\t"
            "sub %2, %%edx\n\t" // substract cycle count
            "sbb %3, %%eax"     // substract cycle count
            : "=&a" (cycles_used)
            , "=&d" (edx)
            , "=&r" (temp1)
            , "=&r" (temp2)
            : "m" (address)
            );
        if (cycles_used < max_latency)
            latencies[cycles_used]++;
        else 
            latencies[max_latency - 1]++;
    }

    // munmap(random_data, lower_cache_size);

    return 0;
} 

int cache_latency(double* address)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    sched_setaffinity(0, sizeof(cpu_set_t), &set);

    size_t cache_sizes[32];
    int num_data_caches = i386_cpuid_caches(cache_sizes);

    int latencies[0x400];
    memset(latencies, 0, sizeof(latencies));

    // Get empty cycles (= overhead??)
    int empty_cycles = 0;

    int i;
    int attempts = 1000000;
    for (i = 0; i < attempts; i++) { // measure how much overhead we have for counting cycles
        int32_t cycles_used, edx, temp1, temp2;
        asm (
            "mfence\n\t"        // memory fence
            "rdtsc\n\t"         // get cpu cycle count
            "mov %%edx, %2\n\t"
            "mov %%eax, %3\n\t"
            "mfence\n\t"        // memory fence
            "mfence\n\t"
            "rdtsc\n\t"
            "sub %2, %%edx\n\t" // substract cycle count
            "sbb %3, %%eax"     // substract cycle count
            : "=&a" (cycles_used)
            , "=&d" (edx)
            , "=&r" (temp1)
            , "=&r" (temp2)
            :
            );
        if (cycles_used < sizeof(latencies) / sizeof(*latencies))
            latencies[cycles_used]++;
        else 
            latencies[sizeof(latencies) / sizeof(*latencies) - 1]++;
    }

    // Calculate empty cycles(overhead)
    {
        // Max Count trial
        int j;
        int maxIdx = 0;
        int maxVal = 0;
        for (j = 0; j < sizeof(latencies) / sizeof(*latencies); j++) {
            if (latencies[j] > maxVal) {
                maxVal = latencies[j];
                maxIdx = j;
            }
        }
        empty_cycles = maxIdx;
        fprintf(stderr, "Empty counting takes %d cycles\n", empty_cycles);
    }

    memset(latencies, 0, sizeof(latencies));
    int cache_level = 3;
    test_cache(attempts, cache_sizes[cache_level-1] / 4, address, latencies, sizeof(latencies) / sizeof(*latencies));

    // Max Count trial
    int j;
    int maxIdx = 0;
    int maxVal = 0;
    for (j = 0; j < sizeof(latencies) / sizeof(*latencies); j++) {
        if (latencies[j] > maxVal) {
            maxVal = latencies[j];
            maxIdx = j;
        }
    }
    fprintf(stderr, "Address %p LLC access takes %d cycles\n", address, maxIdx - empty_cycles);

    return 0;
}