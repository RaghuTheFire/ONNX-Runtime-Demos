# Onnx Runtime Installation

- Install Required Dependencies:
sudo apt-get install -y libprotobuf-dev protobuf-compiler libcurl4-openssl-dev
sudo apt-get install -y libssl-dev libgtest-dev python3-dev python3-setuptools

- Install Python Packages:
pip3 install numpy

- Set Up Environment Variables for CUDA and cuDNN:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

- Configure and Build ONNX Runtime:
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda --build_shared_lib

- Install the Built Libraries:
After the build completes, the libraries and headers will be available in the build/Linux/Release directory.
You can copy these to your system directories or use them directly in your C++ project.

- Compilation
g++ -o test_onnxruntime test_onnxruntime.cpp -I<path_to_onnxruntime_include> -L<path_to_onnxruntime_lib> -lonnxruntime -lonnxruntime_providers_shared -lcudart -lprotobuf -pthread



