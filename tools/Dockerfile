FROM nvcr.io/nvidia/deepstream:5.1-21.02-devel

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install torch torchvision
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd torch2trt && python3 setup.py install
RUN pip3 install tqdm cython pycocotools
RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose
RUN cd trt_pose && python3 setup.py install
RUN pip3 install --upgrade pip
RUN pip install opencv-python pandas Pillow scikit-learn nvidia-tensorrt==7.2.* --index-url https://pypi.ngc.nvidia.com tensorflow
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get upgrade libcudnn8 -y
COPY ./libnvds_osd_cuda11.so /opt/nvidia/deepstream/deepstream-5.1/lib/
