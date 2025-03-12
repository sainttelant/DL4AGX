# build docker
docker build --build-arg TMPDIR=/home/wilson/disklarge/tmp -t uniad_wilson_jetson -f uniad_torch1.12.dockerfile .

2.# docker run 
docker run --privileged -it --rm --runtime nvidia --shm-size=16g --gpus all  -v /home/nvidia/disklarge:/workspace wilson_orin_uniad_v1.2.2 /bin/bash

2.1 docker run in tensorrt8.6
docker run --privileged -it --rm --runtime nvidia --shm-size=16g --gpus all  -v /home/nvidia/disklarge:/workspace uniad_tensorrt8.6 /bin/bash


3. # close firewall in workstation, and enable orin could access it
in workstation install samba
sudo apt update
sudo apt install samba

3.1.config smb
sudo nano /etc/samba/smb.conf
 
[shared]
path = /home/Cnworkspace
available = yes
valid users = wilsxue
read only = no
browsable = yes
public = no
writable = yes
hosts allow = 10.234.32.67


sudo mount -t cifs -o username=wilsxue,password=1,sec=ntlmv2 //10.234.32.12/shared /home/nvidia/disklarge/shared/

3.2. disable firewall in workstation
sudo ufw disable 
or sudo ufw allow 139/tcp  sudo ufw allow 445/tcp

3. #create softlinks from shared folders
ln -s /workspace/shared/nuscenes/ /workspace/UniAD/data/


# build engine tensorrt

# build in host machine

# build tensorrt plugin in folder of inference_app
cmake .. -DTENSORRT_PATH=/usr/include/aarch64-linux-gnu/ && make -j4


MIN=901
OPT=901
MAX=1150
DAT_PATH= /home/wilsxue/Bev/DL4AGX/AV-Solutions/uniad-trt/UniAD/nuscenes_np/uniad_trt_input
SHAPES="prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}"
INPUTS="max_obj_id:${DAT_PATH}/max_obj_id.dat,img_metas_can_bus:${DAT_PATH}/img_metas_can_bus.dat,img_metas_lidar2img:${DAT_PATH}/img_metas_lidar2img.dat,img:${DAT_PATH}/img.dat,use_prev_bev:${DAT_PATH}/use_prev_bev.dat,prev_bev:${DAT_PATH}/prev_bev.dat,command:${DAT_PATH}/command.dat,timestamp:${DAT_PATH}/timestamp.dat,l2g_r_mat:${DAT_PATH}/l2g_r_mat.dat,l2g_t:${DAT_PATH}/l2g_t.dat,prev_track_intances0:${DAT_PATH}/prev_track_intances0.dat,prev_track_intances1:${DAT_PATH}/prev_track_intances1.dat,prev_track_intances3:${DAT_PATH}/prev_track_intances3.dat,prev_track_intances4:${DAT_PATH}/prev_track_intances4.dat,prev_track_intances5:${DAT_PATH}/prev_track_intances5.dat,prev_track_intances6:${DAT_PATH}/prev_track_intances6.dat,prev_track_intances8:${DAT_PATH}/prev_track_intances8.dat,prev_track_intances9:${DAT_PATH}/prev_track_intances9.dat,prev_track_intances11:${DAT_PATH}/prev_track_intances11.dat,prev_track_intances12:${DAT_PATH}/prev_track_intances12.dat,prev_track_intances13:${DAT_PATH}/prev_track_intances13.dat,prev_timestamp:${DAT_PATH}/prev_timestamp.dat,prev_l2g_r_mat:${DAT_PATH}/prev_l2g_r_mat.dat,prev_l2g_t:${DAT_PATH}/prev_l2g_t.dat"
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH \
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/nvidia/disklarge/DL4AGX/AV-Solutions/uniad-trt/onnx/uniad_tiny_dummy.onnx \
  --saveEngine=/home/nvidia/disklarge/DL4AGX/AV-Solutions/uniad-trt/engine/out.engine \
  --plugins=/home/nvidia/disklarge/DL4AGX/AV-Solutions/uniad-trt/inference_app/build/libuniad_plugin.so\
  --verbose \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --useCudaGraph \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --loadInputs=${INPUTS}



# build engine tensorrt in docker
MIN=901
OPT=905
MAX=1150
DAT_PATH= /workspace/DL4AGX/AV-Solutions/uniad-trt/UniAD/dumped_inputs
SHAPES="prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}"
INPUTS="max_obj_id:${DAT_PATH}/max_obj_id.dat,img_metas_can_bus:${DAT_PATH}/img_metas_can_bus.dat,img_metas_lidar2img:${DAT_PATH}/img_metas_lidar2img.dat,img:${DAT_PATH}/img.dat,use_prev_bev:${DAT_PATH}/use_prev_bev.dat,prev_bev:${DAT_PATH}/prev_bev.dat,command:${DAT_PATH}/command.dat,timestamp:${DAT_PATH}/timestamp.dat,l2g_r_mat:${DAT_PATH}/l2g_r_mat.dat,l2g_t:${DAT_PATH}/l2g_t.dat,prev_track_intances0:${DAT_PATH}/prev_track_intances0.dat,prev_track_intances1:${DAT_PATH}/prev_track_intances1.dat,prev_track_intances3:${DAT_PATH}/prev_track_intances3.dat,prev_track_intances4:${DAT_PATH}/prev_track_intances4.dat,prev_track_intances5:${DAT_PATH}/prev_track_intances5.dat,prev_track_intances6:${DAT_PATH}/prev_track_intances6.dat,prev_track_intances8:${DAT_PATH}/prev_track_intances8.dat,prev_track_intances9:${DAT_PATH}/prev_track_intances9.dat,prev_track_intances11:${DAT_PATH}/prev_track_intances11.dat,prev_track_intances12:${DAT_PATH}/prev_track_intances12.dat,prev_track_intances13:${DAT_PATH}/prev_track_intances13.dat,prev_timestamp:${DAT_PATH}/prev_timestamp.dat,prev_l2g_r_mat:${DAT_PATH}/prev_l2g_r_mat.dat,prev_l2g_t:${DAT_PATH}/prev_l2g_t.dat"
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/lib:$LD_LIBRARY_PATH \
/usr/src/tensorrt/bin/trtexec \
  --onnx=/workspace/DL4AGX/AV-Solutions/uniad-trt/onnx/uniad_tiny_imgx0.25_cp.onnx \
  --saveEngine=/workspace/DL4AGX/AV-Solutions/uniad-trt/engine/out.engine \
  --plugins=/workspace/DL4AGX/AV-Solutions/uniad-trt/inference_app/build/libuniad_plugin.so\
  --verbose \
  --dumpLayerInfo \
  --dumpProfile \
  --exportLayerInfo=layinfor.txt \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --useCudaGraph \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --loadInputs=${INPUTS}


  # inference under folder of uniad-trt
  
  run :
  LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH ./inference_app/build/uniad engine/out.engine \
  ./inference_app/build/libuniad_plugin.so UniAD/nuscenes_np/uniad_trt_input/ results/ 1000






