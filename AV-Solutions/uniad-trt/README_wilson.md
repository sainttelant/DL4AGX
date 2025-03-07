# run docker

docker run -it --gpus all --shm-size=16g -v /home/wilsxue/Bev:/workspace/Bev -v /home/Cnworkspace/nuscenes:/workspace/nuscenes uniad_torch1.12_tensorrt_v0.2 /bin/bash

# analysis lib library in TensorRt build

ldd libtensorrt_ops.so  will show which so will be depended. 
if erroring not found , you need to export something lib in working environ

nm -D libtensorrt_ops.so | grep grid_sample  : that will check wheather all symbols have already compiled or linked to the library of grid_sample. if not ,you should check the original srcs and CMAKELISTS, notice, CXXCUDA language is important.

and to see the symbol in c++ , you should type like

echo '_Z22grid_sampler_2d_kernelI6__halfEviPKT_S3_PS1_N6helper10TensorDescES6_S6_24GridSamplerInterpolation18GridSamplerPaddingb' | c++filt



