SHELL = /bin/sh
CC    = gfortran
FLAGS = -std=f2018
# CFLAGS = -fPIC -Wall -Werror -pedantic
CFLAGS = -Wall -Werror -pedantic -I/usr/include
# LDFLAGS = -shared -lm -lgsl -lgslcblas -ltsnnls
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/lapack -llapack -lblas
DEBUGFLAGS = -g -g3 -O0 -ggdb3
RELEASEFLAGS = -O2

TARGET = nnls
SOURCES = nnls.f90 main.f90
OBJECTS = $(SOURCES:.c=.o)

.PHONY: clean

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)
