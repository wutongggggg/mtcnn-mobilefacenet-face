# 人脸检测与识别 · MTCNN + MobileFaceNet

基于 **PyTorch** 复现 **MTCNN 人脸检测 + MobileFaceNet 人脸识别**，支持图片与摄像头实时识别。课程《人工智能》上机实习项目。

## 功能
- MTCNN 人脸检测（PNet / RNet / ONet）
- MobileFaceNet 人脸特征提取与比对识别
- 支持图片批量识别与摄像头实时识别

## 技术栈
Python、PyTorch、OpenCV、MTCNN、MobileFaceNet

## 目录
- `src/train.py` 训练
- `src/eval.py` 评估
- `src/infer.py` 图片推理；`src/infer_camera.py` 摄像头实时识别
- `src/create_dataset.py` 数据集构建
- `src/net/` 网络结构；`src/utils/` 数据与工具
- `docs/AI实习报告.pdf` 上机实习报告

## 运行
```bash
pip install -r requirements.txt
python src/infer_camera.py      # 需自备模型权重（见报告说明）
```

## 说明
本项目为课程上机实习，基于公开实现复现；模型权重文件未随仓库提供，可按报告中流程训练或自行准备。
