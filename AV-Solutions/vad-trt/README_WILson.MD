# build the docker 

docker build -t vad_torch1.12_x86 -f vad_torch1.12.1.dockerfile .

docker build -t vad_official_x86 -f dockerfile .

# docker run

docker run --privileged -it --gpus all --shm-size=16g -v /home/wilsxue.adm/Bev:/workspace/Bev -v /data_fast1/nuscenes:/workspace/Bev/VAD/data --name wilson_vad vad_torch1.12_x86_v1.0 /bin/bash

# official docker

docker run --name=vad_official_x86 -d -it --rm --shm-size=4096m --privileged --gpus all -it --network=host \
   -v /home/wilsxue.adm/Bev:/workspace/Bev -v /data_fast1/nuscenes:/workspace/Bev/VAD/data \
   vad_official_x86 /bin/bash
# link data in workstation of docker inside




# convert to ONNX
in /workspace/Bev/DL4AGX/AV-Solutions/vad-trt/export_eval



export PYTHONPATH=.:/workspace/Bev/VAD

# it is very important to change config of img in VAD_tiny_stage_2.py
img_norm_cfg = dict(
  mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# 转onnx的时候必须使用vad_official_x86 的container，否则有错误
step one: python export_no_prev.py /workspace/Bev/VAD/projects/configs/VAD/VAD_tiny_stage_2.py /workspace/Bev/VAD/ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

step two: python export_prev.py /workspace/Bev/VAD/projects/configs/VAD/VAD_tiny_stage_2.py /workspace/Bev/VAD/ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

# tensorrt 
# build tensorrt plugin
export TRT_ROOT=/usr/include/x86_64-linux-gnu/
cd /workspace/Bev/DL4AGX/AV-Solutions/vad-trt/plugins/build 

# set build target in cmakelist.txt 
cmake ..


# prepare export environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/x86_64-linux-gnu/
export PATH=$PATH:/usr/bin

# save engine file
trtexec --onnx=scratch/vadv1.extract_img_feat/sim_vadv1.extract_img_feat.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --profilingVerbosity=detailed --dumpProfile \
        --separateProfileRun --useSpinWait --useManagedMemory \
        --saveEngine=vadv1.extract_img_feat/vadv1.extract_img_feat.fp32.engine
        --verbose

trtexec --onnx=scratch/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.onnx \
        --staticPlugins=../plugins/build/libplugins.so \
        --profilingVerbosity=detailed --dumpProfile \
        --separateProfileRun --useSpinWait --useManagedMemory \
        --saveEngine=vadv1.pts_bbox_head.forward/vadv1.pts_bbox_head.forward.engine
        --verbose

# save data executed in the folder of export_eval
ln -s /workspace/Bev/VAD/data/ data

python save_data.py /workspace/Bev/VAD/projects/configs/VAD/VAD_tiny_stage_2.py /workspace/Bev/VAD/ckpts/VAD_tiny.pth --launcher none --eval bbox --tmpdir tmp

