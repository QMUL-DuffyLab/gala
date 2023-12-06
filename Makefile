SHELL = /bin/sh
CC    = gcc
FLAGS = -std=c11
CFLAGS = -fPIC -Wall -Werror -pedantic
LDFLAGS = -shared -lm -lgsl -lgslcblas
DEBUGFLAGS = -g -ggdb3
RELEASEFLAGS = -O2

TARGET = libantenna.so
SOURCES = antenna.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: clean

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(FLAGS) $(CFLAGS) $(LDFLAGS) $(RELEASEFLAGS) -o $(TARGET) $(OBJECTS)