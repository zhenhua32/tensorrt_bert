[TOC]

# 使用 tensorrt 的实践指南

# torch 版本

[官方文档](https://pytorch.org/TensorRT/index.html)

[使用 bert 的示例 notebook 笔记本](https://github.com/pytorch/TensorRT/blob/master/notebooks/Hugging-Face-BERT.ipynb)

[个人实践的 notebook 笔记本](./torch/%E5%AE%9E%E8%B7%B5_torch_bert.ipynb)

[nvidia-pytorch 镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

当前使用的镜像是 nvcr.io/nvidia/pytorch:22.05-py3, 在最新的 nvcr.io/nvidia/pytorch:22.11-py3 没有跑成功.

实验环境是在 windows 的 WSL2 上跑的. 目前看起来 docker 还是挺香的, 不用考虑装驱动和环境的. 唯一的缺点是太耗硬盘了.

```bash
docker pull nvcr.io/nvidia/pytorch:22.11-py3
docker run --rm -it --gpus all nvcr.io/nvidia/pytorch:22.11-py3 bash
nvidia-smi
```

# tensorflow 1.x 版本

TODO: 这个属实难崩.

# tensorflow 2.x 版本

# ONNX

[在 GPU 上使用 onnxruntime 的 bert 示例](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)

总之, 应该先把模型导出成 ONNX 格式.

tensorflow 可以使用 [tf2onnx](https://github.com/onnx/tensorflow-onnx).
torch 可以使用自带的 [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export).

# ONNX + tensorrt

环境没装过, 可能比较复杂.



