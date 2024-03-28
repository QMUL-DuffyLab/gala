SHELL = /bin/sh
PREFIX = $(shell echo $$CONDA_PREFIX)
$(info $(PREFIX))
FC    = gfortran
FLAGS = -std=f2018 -ffree-form
FFLAGS = -Wall -Werror -pedantic -fcheck=all -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -llapack -lblas -ljsonfortran
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

DEBUG = 1
ifeq ($(DEBUG), 1)
	DFLAGS = -g -g3 -O0 -ggdb3
else
	DFLAGS = -O2
endif

OBJECTS = $(patsubst %.f, %.o, $(SOURCES))

$(OBJECTS): %.o : %.f
	$(FC) $(FLAGS) $(DFLAGS) $(FFLAGS) -c -o $@ $<

$(TARGET): $(OBJECTS)
	$(FC) $(FLAGS) $(FFLAGS) $(DFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

.PHONY: all clean check

all: $(TARGET)

check:
	@echo "SOURCES = $(SOURCES)"
	@echo "OBJECTS = $(OBJECTS)"
	@echo "$(FC) $(FLAGS) $(FFLAGS) -o $@ $<"


clean:
	rm -f $(OBJECTS) $(TARGET) $(patsubst %.o, %.mod, $(OBJECTS))
