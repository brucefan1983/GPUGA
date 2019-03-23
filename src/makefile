CC = nvcc
CFLAGS = -std=c++11 -O3 -arch=sm_35 -DDEBUG
SOURCES = main.cu gpuga.cu ga.cu fitness.cu read_file.cu error.cu
_OBJ = main.o gpuga.o ga.o fitness.o read_file.o error.o
HEADERS = gpuga.cuh ga.cuh fitness.cuh read_file.cuh error.cuh
ODIR = obj
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
all: gpuga
$(ODIR)/%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@
gpuga: $(OBJ)
	$(CC) $^ -o $@
clean:
	rm $(ODIR)/*o gpuga

