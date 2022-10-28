#define _GNU_SOURCE

#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

void print_pagemap_entry(unsigned long long pagemap_entry);
unsigned long long get_pagemap_entry(void *va);

unsigned long long get_pagemap_entry(void *va)
{
    ssize_t ret;
    off_t myoffset;

    pid_t mypid;
    char filename[32];
    unsigned long long result;
    static int pagemap_fd;
    static int initialized = 0;

    if (initialized == 0) {
        mypid = getpid();
        printf("mypid %d\n", mypid);
        sprintf(filename, "/proc/%d/pagemap", mypid);
        pagemap_fd = open(filename, O_RDONLY);
        if (pagemap_fd == 0) {
            return(0UL);
        }
        initialized = 1;
    }

    myoffset = ((long long) va >> 12) << 3;

    ret = pread(pagemap_fd, &result, 8, myoffset);
    if (ret != 8) {
        return(0UL);
    }

    return result;
}

#define BIT_IS_SET(x, n)    (((x) & (1UL<<(n))) ? 1 : 0)

void print_pagemap_entry(unsigned long long pagemap_entry)
{
    int logpagesize;
    int pagesize;
    unsigned long framenumber;
    unsigned long tmp;

    // https://www.kernel.org/doc/Documentation/vm/pagemap.txt for more details about pagemap
    tmp = BIT_IS_SET(pagemap_entry, 63);
    if (tmp == 0) {
        printf("WARNING in print_pagemap_entry: page is not present. Result = %.16llx\n", pagemap_entry);
    }
    tmp = BIT_IS_SET(pagemap_entry, 62);
    if (tmp != 0) {
        printf("WARNING in print_pagemap_entry: page is swapped. Result = %.16llx\n", pagemap_entry);
    }

    framenumber = ((pagemap_entry << 9) >> 9);

    printf("print_pagemap_entry: argument = 0x%.16llx, framenumber = 0x%.16lx\n",pagemap_entry,framenumber);

    // logpagesize = ( (pagemap_entry<<3) >> 58);	// clear bits 61-63, then shift (original) bit 55 down to 0;
	// pagesize = 1 << logpagesize;
	// printf("print_pagemap_entry: logpagesize = %d, pagesize = %d, framenumber = 0x%.16lx\n",logpagesize,pagesize,framenumber);
}