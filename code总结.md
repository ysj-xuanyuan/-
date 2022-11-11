# 超分
```bash
python inference_realesrgan.py -n RealESRNet_x2plus -i  [img_dir]  -o [output_dir] --model_path experiments/pretrained_models/net_g_950000.pth

python inference_realesrgan.py -n RealESRNet_x2plus -i  [img_dir]  -o ./tal_videos_result/demo_ysj/ --model_path experiments/pretrained_models/net_g_950000.pth

name=$1
num=$2
ffmpeg -i tal_videos/0805/${name}.mp4 -f mp3 -vn delogo_out/${name}.mp3 #提取音频保存为MP3
python video2frame.py tal_videos/0805/${name}.mp4   tal_videos_result/${name} #视频抽帧为图片

CUDA_VISIBLE_DEVICES=${num} python inference_realesrgan.py -n RealESRNet_x2plus -i tal_videos_result/${name} -o delogo_out/${name} --model_path ./experiments/pretrained_models/net_g_950000.pth #推理生成超分辨率结果

ffmpeg -r 25 -i delogo_out/${name}/%8d_out.png -vf fps=25 -vcodec h264 -b:v 3000k delogo_out/${name}.mp4 # 将图片合成为视频
ffmpeg -i delogo_out/${name}.mp4 -i delogo_out/${name}.mp3 -c copy delogo_out/${name}_SR.mp4 # 视频与声音合并
```



# 人脸

```bash
python inference_gfpgan.py -i  [img_dir]  -o [output_dir] -v 1.3 -s 1 --bg_upsampler None 
python inference_gfpgan.py -i  inputs/whole_imgs -o results -v 1.3 -s 1 --bg_upsampler None 
```



VScode编辑器



# ffmpeg

## 主要参数格式

-i 设置输入流（input）

-f 设置输出格式（format）

-ss 开始时间

-t 时间长度

## 视频参数

-vframe  设置输出的视频帧数

-b 设置视频码率

-b:v 视频码率

-r 视频帧速率

-s 设置画面的宽与高

-vn 去除视频流

-aspect aspect设置横纵比4:3  16:9

-vcode 设置视频编解码器

-vf 设置视频过滤器 可以有 fps=25 或者 scale=480:-1

## 音频参数

-aframes

```bash
# sample
ffmpeg -r 25 -i delogo_out/${name}/%8d_out.png -vf fps=25 -vcodec h264 -b:v 3000k delogo_out/${name}.mp4 # 将图片合成为视频
```



# tensorRT

TensorRT是可以在**NVIDIA**各种**GPU硬件平台**下运行的一个**C++推理框架**。我们利用Pytorch、TF或者其他框架训练好的模型，可以转化为TensorRT的格式，然后利用TensorRT推理引擎去运行我们这个模型，从而提升这个模型在英伟达GPU上运行的速度。速度提升的比例是**比较可观**的。





# Pytorch模型转成ONNX和MNN

参考文章：[(36条消息) Pytorch模型转成ONNX和MNN_忘泪的博客-CSDN博客](https://blog.csdn.net/wl1710582732/article/details/107743268)
## PyTorch转ONNX

需要安装好pytorch环境和onnx包

```bash
pip install onnx
```

以mobilenet为例，下载好[mobilenet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py)和预训练模型[mobilenet_v2-b0353104.pth](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)，转换代码如下

```python 
import torch
import torch.nn as nn
import torch.onnx
import onnx
from mobilenet import mobilenet_v2

pt_model_path = './mobilenet_v2-b0353104.pth'
onnx_model_path = './mobilenet_v2-b0353104.onnx'
model = mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load(pt_model_path,map_location=torch.device('cpu')))
input_tensor = torch.randn(1, 3, 224, 224)
input_names = ['input']
output_names = ['output']
torch.onnx.export(model, input_tensor, onnx_model_path, verbose=True, input_names=input_names, output_names=output_names)

```

例2

```python
import os.path as osp
 
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
 
# your test input array
test_arr = np.random.randn(1, 3, 224, 224).astype(np.float32)

# torch model
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = torchvision.models.resnet50(pretrained=True).cuda().eval()
print('pytorch result:', model(torch.from_numpy(test_arr).cuda()))
 
input_names = ["input"]
output_names = ["output"]
 
if not osp.exists('resnet50.onnx'):
    # translate your pytorch model to onnx
    torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True, input_names=input_names, output_names=output_names)
 
model = onnx.load("resnet50.onnx")
ort_session = ort.InferenceSession('resnet50.onnx')
outputs = ort_session.run(None, {'input': test_arr})

print('onnx result:', outputs[0])
```




# ONNX

测试 onnx runtime

```python
import onnx
import numpy as np
import onnxruntime as rt
import cv2

model_path = '/home/oldpan/code/models/Resnet34_3inputs_448x448_20200609.onnx'

# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# 读入图像并调整为输入维度
image = cv2.imread("data/images/person.png")
image = cv2.resize(image, (448,448))
image = image.transpose(2,0,1)
image = np.array(image)[np.newaxis, :, :, :].astype(np.float32)

# 设置模型session以及输入信息
sess = rt.InferenceSession(model_path)
input_name1 = sess.get_inputs()[0].name
input_name2 = sess.get_inputs()[1].name
input_name3 = sess.get_inputs()[2].name

output = sess.run(None, {input_name1: image, input_name2: image, input_name3: image})
print(output)
```

