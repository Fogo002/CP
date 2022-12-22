CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
THREADS = 38
CFLAGS = -O3 -fopenmp -g -fno-omit-frame-pointer -Wall -funroll-loops -ftree-vectorize -msse4 
.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC) -lm

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

runseq:
	./$(BIN)$(EXEC) 1048576 $(CP_CLUSTERS)

runpar:
	./$(BIN)$(EXEC) 1048576 $(CP_CLUSTERS) $(THREADS)
