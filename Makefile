CXX = g++
CXXFLAGS = -I/path_to_onnxruntime_include -L/path_to_onnxruntime_lib -lonnxruntime -lonnxruntime_providers_shared -lcudart -lprotobuf -pthread

TARGET = test_onnxruntime
SRCS = test_onnxruntime.cpp

$(TARGET): $(SRCS)
    $(CXX) -o $(TARGET) $(SRCS) $(CXXFLAGS)

clean:
    rm -f $(TARGET)
