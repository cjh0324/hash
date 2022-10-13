CC = gcc
LINK_OPTION = -lz
CFLAGS = -g
OBJS=hash.o va2pa_lib.o cache_latency.o polling.o
TARGET=hash.out


$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)

hash.o: hash.c
va2pa_lib.o: va2pa_lib.c
cache_latency.o: cache_latency.c
polling.o: polling.c

clean:
	rm -f *.o *.out hash
