SHELL = /bin/sh
HOME = $(shell echo $$HOME)
$(info $(HOME))
CC    = gfortran
FLAGS = -std=f2018 -ffree-form
CFLAGS = -Wall -Werror -pedantic -fcheck=all -I$(HOME)/anaconda3/include
LDFLAGS = -L$(HOME)/anaconda3/lib -llapack -lblas -ljsonfortran
DFLAGS = -g -g3 -O0 -ggdb3
RFLAGS = -O2
SOURCES = nnls.f antenna.f

# if SHARED = 1 compile nnls module as a shared library
# if SHARED = 0 compile standalone program to debug
SHARED = 0
ifeq ($(SHARED), 1)
	CFLAGS += -fPIC
	LDFLAGS += -shared
	TARGET = libnnls.so
else
	SOURCES += main.f
	TARGET = antenna_f_test
endif
OBJECTS = $(SOURCES)

.PHONY: clean

all: $(TARGET)

# clean:
# 	rm -f $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(FLAGS) $(CFLAGS) $(DFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)
