SHELL = /bin/sh
CC    = gfortran
FLAGS = -std=f2018
# CFLAGS = -fPIC -Wall -Werror -pedantic
CFLAGS = -Wall -Werror -pedantic -I/usr/include
# LDFLAGS = -shared -lm -lgsl -lgslcblas -ltsnnls
LDFLAGS = -llapack -lblas
DFLAGS = -g -g3 -O0 -ggdb3
RFLAGS = -O2

TARGET = nnls
SOURCES = nnls.f90 main.f90
OBJECTS = $(SOURCES:.c=.o)

.PHONY: clean

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(FLAGS) $(CFLAGS) $(RFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)
