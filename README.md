# OnnxDeployLab

学习 C++、ONNX Runtime 与 YOLO 从 ONNX 模型到生产部署的完整流程。

## 当前进度（2026-10-31）

### 分类闭环

- C++17 / ORT 1.18.0 / OpenCV 4.8.1，默认CPU推理。
- models/cls/hezhuang5.onnx：输入images [-1,3,-1,-1]，输出output0 [-1,28]。
- 当前分类只接受224×224三通道图片：BGR→RGB、float32 /255、HWC→CHW。
- Top-1、28类标签映射、分阶段计时完成，分类算法拆为独立模块。
- 同一个Session依次处理两张图：petxiao/0.697109、you/0.776489；与Python ORT 1.23.2基本一致。

### 分割实验

- models/seg/hei_huizong_250428.onnx：YOLO11n-seg，单类别bottle。
- 2048×1024图片LetterBox到640×640：scale=0.3125，上下各160像素padding。
- 实际输出：[1,37,8400]候选数据、[1,32,160,160]原型图。
- 修复candidateCount应取shape[2]的问题。
- 实验阈值0.25，NMS阈值0.45：8400候选→10→1，保留原始索引3399，分数0.879162。
- 最高分框反算原图位置已在此前阶段目视确认。
- 当前代码已生成160×160的mask概率预览，尚未框裁剪、去padding、还原原图或二值化。

下一次先复习索引、系数和proto，不急着增加新代码。

## 构建与运行

用VS2022打开本目录，沿用CMake/Ninja工作流，选择x64-Debug或x64-Release。
依赖路径默认指向原项目dll目录，通过ORT_ROOT和OpenCV_DIR可调整。
本工程针对当前Windows环境配置，不包含Linux部署配置。
CMake为每个目标合并配置头文件、库及DLL复制。

- model_inspect：分类验证。无参数时处理data/cls_test.jpg和data/cls_test2.jpg。
- inspect_seg：分割学习入口。读取data/seg_test.jpg，保存LetterBox和mask概率预览。

分类也保留可选参数：
```text
model_inspect.exe <分类ONNX路径> <224x224图片路径>
```

标签从模型同目录的labels.txt读取。模型和图片需自行放在本机已确认路径。

## 文件结构

```text
include/Classification.h
src/main.cpp
src/Classification.cpp
src/inspect_seg.cpp
python/check_cls.py
python/check_seg.py
models/cls/labels.txt
docs/
```

Python check_cls.py保存ONNX的标签，可选传图片路径验证第二张图；check_seg.py目前只读取元信息，不是分割数值对齐脚本。

分割产物在data/：
- seg_letterbox.jpg：640×640预处理预览。
- seg_mask_proto_3399.png：当前样本未裁剪的160×160概率预览。
- seg_best_box.jpg为早期单框验证产物，当前多候选代码不再生成该文件。

模型、图片、生成图、DLL和构建目录由.gitignore排除，未上传。
本轮未接入Qt、数据库、设备通信和分选业务。

## 验证边界与归档来源

Windows用户运行日志证明了上述学习步骤；本次仓库代码根据对话片段整合，并非直接读取Windows最新文件。
本环境缺少CMake、OpenCV/ORT开发依赖及模型，未重新编译或执行C++；整理版需要本机重新构建回归。
不将单图成功视为分割数值全面对齐或生产部署完成。

## 学习记录

- [10月29日：分类推理与生命周期](docs/2026-10-29.md)
- [10月30日：分类封装、标签、计时与模块拆分](docs/2026-10-30.md)
- [10月31日：分割索引、LetterBox、NMS与mask概率图](docs/2026-10-31.md)

10月29日记录保留历史状态；当前进度以本README及10月31日记录为准。
