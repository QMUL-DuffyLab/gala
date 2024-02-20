SHELL = /bin/sh
CC    = gfortran
FLAGS = -std=f2018 -ffree-form
CFLAGS = -Wall -Werror -pedantic
LDFLAGS = -llapack -lblas
DFLAGS = -g -g3 -O0 -ggdb3
RFLAGS = -O2
SOURCES = nnls.f

# if SHARED = 1 compile nnls module as a shared library
# if SHARED = 0 compile standalone program to debug
SHARED = 0
ifeq ($(SHARED), 1)
	CFLAGS += -fPIC
	LDFLAGS += -shared
	TARGET = libnnls.so
else
	SOURCES += main.f
	TARGET = nnls
endif
OBJECTS = $(SOURCES:.c=.o)

.PHONY: clean

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(FLAGS) $(CFLAGS) $(RFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)
