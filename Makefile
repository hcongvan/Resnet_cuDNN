CC = g++
CFLAGS = -std=c++11 -g
CC_VERSION = 5.4.0

CUDA_PATH ?= /usr/local/cuda
PROJ_PATH = /home/ubuntu/VideoDecode
STANDARD_LIB = /usr/include/c++/$(CC_VERSION)
LIB_PATH = /usr/lib/x86_64-linux-gnu
INC_PATH = /usr/include/x86_64-linux-gnu

INCLUDES = -I$(CUDA_PATH)/include
INCLUDES += -I$(PROJ_PATH)
INCLUDES += -I$(PROJ_PATH)/NvCodec
INCLUDES += -I$(INC_PATH)
INCLUDES += -I$(STANDARD_LIB)

LFLAGS = -L$(CUDA_PATH)/lib64/stubs
LFLAGS += -L$(CUDA_PATH)/lib64
LFLAGS += -L$(PROJ_PATH)/NvCodec/Lib
LFLAGS += -L$(LIB_PATH)

#LIBS = -lsstream -lmutex -lvector -liostream -lalgorithm -lnvcuvid
#LIBS = $(shell pkg-config --libs libavcodec libavutil libavformat)
LIBS += -lcuda -lnvcuvid -lavformat -lavutil -lswresample -lm -lz -lavcodec -lswscale

SRCS = Resnet.cpp VideoDecode.cpp 

OBJS = $(SRCS:.c=.o)
MAIN = Resnet

.PHONY: depend clean

all: $(MAIN)

VideoDecode.o : VideoDecode.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LFLAGS) $(LIBS)
Resnet.o : Resnet.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $< $(LFLAGS) $(LIBS)
Resnet : VideoDecode.o Resnet.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS) $(LIBS) 

clean:
	rm -rf *.o *~ $(MAIN)
depend: $(SRCS)
	makedepend $(INCLUDES) $^

