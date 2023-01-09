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

TODO: `torch.onnx.export` 可能很慢. 

# ONNX + tensorrt

环境没装过, 可能比较复杂.

底层的代码用的是 NVIDIA 官方的 python 包 [tensorrt](https://github.com/NVIDIA/TensorRT).
但不要信文档上写的 `pip install tensorrt`.
API 文档参考这里 [tensorrt python api](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/coreConcepts.html).

这个 NVIDIA 官方的 github 仓库有个工具挺使用的. 转换 ONNX 模型到 trt 模型就是靠它.
[Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools#converting-a-model-to-tensorrt)

直接调用 tensorrt 的代码可能会有点复杂, 可以参考这个仓库, 做了一些封装.
[transformer-deploy](https://github.com/ELS-RD/transformer-deploy/tree/v0.4.0)
