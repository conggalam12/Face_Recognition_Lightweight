.PHONY: all clean

CXX = g++
FLAGS = -I/home/congnt/congnt/VNPT/Face_Recognition_ncnn/include/ncnn
TARGET = run

LIB_DIR = /home/congnt/congnt/VNPT/Face_Recognition_ncnn/lib

OPENCV = `pkg-config opencv4 --cflags --libs`

OBJ_DIR = obj
SRC_DIR = src

MAIN = $(SRC_DIR)/main.cpp 
LIB_OPENCV = $(OPENCV)

MODEL_DIR = src/model
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/UltraFace.o

LIB += -L$(LIB_DIR)
LIB += -lncnn -fopenmp

DEPS = $(MODEL_DIR)/UltraFace.h

all: $(TARGET)

$(OBJ_DIR)/main.o: $(MAIN) $(DEPS)
	mkdir -p $(OBJ_DIR)
	$(CXX) $(FLAGS) -c -o $@ $< $(LIB_OPENCV) $(LIB)

$(OBJ_DIR)/UltraFace.o: $(MODEL_DIR)/UltraFace.cpp $(DEPS)
	mkdir -p $(OBJ_DIR)
	$(CXX) $(FLAGS) -c -o $@ $< $(LIB) 

$(TARGET): $(OBJS)
	$(CXX) $(FLAGS) -o $@ $^ $(LIB_OPENCV) $(LIB)

clean:
	rm -rf $(TARGET) $(OBJ_DIR)