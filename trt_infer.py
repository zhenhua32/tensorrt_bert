"""
将推理代码重新整理下, 基本就是复刻下 trt_utils.py
主要是可以直接拿来用, 虽然 trt_utils.py 也可以直接拿来用
还是得自己整一遍才印象深刻

环境: 基于 tensorrt 版本 `8.2.5.1`, 目前用的镜像中是该版本的, 没试过别的版本.
官方文档都没有选择版本的地方, 官方文档当前是 `8.5.2` 版本.
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/coreConcepts.html
"""

import logging
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

import tensorrt as trt
import torch
from tensorrt import (
    Builder,
    IBuilderConfig,
    ICudaEngine,
    IElementWiseLayer,
    IExecutionContext,
    ILayer,
    INetworkDefinition,
    IOptimizationProfile,
    IReduceLayer,
    Logger,
    OnnxParser,
    Runtime,
)
from transformers import BertTokenizer


@dataclass
class Input:
    name: str
    min_shape: List[int]
    opt_shape: List[int]
    max_shape: List[int]


def build_engine(
    onnx_file_path: str, inputs: List[Input], max_workspace_size: int, fp16=False, int8=False
) -> ICudaEngine:
    """
    构建 trt engine
    """
    # Logger for the Builder, ICudaEngine and Runtime . 日志服务类
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    # Allows a serialized ICudaEngine to be deserialized. 运行时, 用来承载模型
    runtime: Runtime = trt.Runtime(trt_logger)
    # Builds an ICudaEngine from a INetworkDefinition . 构建器
    builder: Builder = trt.Builder(trt_logger)
    network_def: INetworkDefinition = builder.create_network(
        # 目前 NetworkDefinitionCreationFlag 的参数中就这一个, 另一个 EXPLICIT_PRECISION 已经被弃用且没效果了.
        flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    )

    # 文档中就三个 parser, UFF Parser, Caffe Parser, Onnx Parser
    # This class is used for parsing ONNX models into a TensorRT network definition
    parser: OnnxParser = trt.OnnxParser(network_def, trt_logger)
    # Create a builder configuration object. 构建配置
    config: IBuilderConfig = builder.create_builder_config()

    # 设置模型使用的最大GPU内存
    # WARN: 这个属性在新版被抛弃了
    config.max_workspace_size = max_workspace_size

    # BuilderFlag 有一堆设置, 但其他的还不是很了解
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # 解析模型
    with open(onnx_file_path, "rb") as f:
        parser.parse(model=f.read(), path=onnx_file_path)

    # 创建优化配置, 当前一个就够了, 不用搞多配置
    profile: IOptimizationProfile = builder.create_optimization_profile()
    # 设置形状
    for input in inputs:
        profile.set_shape(
            input=input.name,
            min=input.min_shape,
            opt=input.opt_shape,
            max=input.max_shape,
        )
    # 添加优化配置
    config.add_optimization_profile(profile)

    # 真正开始构建网络
    trt_engine = builder.build_serialized_network(network_def, config)
    # 反序列化网络
    engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)

    assert engine is not None, "创建 engine 失败"
    return engine


def load_engine(trt_file_path: str) -> ICudaEngine:
    """
    加载 trt engine
    """
    # Logger for the Builder, ICudaEngine and Runtime . 日志服务类
    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    # Allows a serialized ICudaEngine to be deserialized. 运行时, 用来承载模型
    runtime: Runtime = trt.Runtime(trt_logger)
    with open(file=trt_file_path, mode="rb") as f:
        # 反序列化, 返回 ICudaEngine. An ICudaEngine for executing inference on a built network.
        engine: ICudaEngine = runtime.deserialize_cuda_engine(f.read())
    return engine


def save_engine(engine: ICudaEngine, trt_file_path: str) -> None:
    """
    保存 engine 到文件中
    """
    with open(trt_file_path, "wb") as f:
        f.write(engine.serialize())


class Infer:
    def __init__(self, engine: ICudaEngine, profile_index: int = 0):
        self.engine = engine
        self.profile_index = profile_index
        self.context, self.input_binding_idxs, self.output_binding_idxs = self.prepare()

    def prepare(self):
        """
        推理前准备工作
        """
        # A CUDA stream is a linear sequence of execution that belongs to a specific device, independent from other streams. See CUDA semantics for details.
        # .cuda_stream 属性好像不在 torch 的文档上. 而且类型是 int
        stream: int = torch.cuda.current_stream().cuda_stream
        # Create an IExecutionContext . 创建一个执行上下文
        # Context for executing inference using an ICudaEngine . Multiple IExecutionContext s may exist for one ICudaEngine instance, allowing the same ICudaEngine to be used for the execution of multiple batches simultaneously.
        context: IExecutionContext = self.engine.create_execution_context()
        # Set the optimization profile with async semantics. 加载优化配置文件
        context.set_optimization_profile_async(profile_index=self.profile_index, stream_handle=stream)
        # retrieve input/output IDs. 获取输入和输出的索引
        input_binding_idxs, output_binding_idxs = self.get_binding_idxs(
            self.engine, self.profile_index
        )
        return context, input_binding_idxs, output_binding_idxs

    @staticmethod
    def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
        """
        根据 profile_index 计算输入和输出的索引
        """
        # num_bindings 就是输入和输出的名字数量. num_optimization_profiles 是优化配置文件的数量
        num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
        # 开始的索引位置
        start_binding = profile_index * num_bindings_per_profile
        # 结束的索引位置
        end_binding = (
            start_binding + num_bindings_per_profile
        )  # Separate input and output binding indices for convenience
        input_binding_idxs: List[int] = []
        output_binding_idxs: List[int] = []
        # 判断每个索引位置是否是输入, 将输入和输出的索引分别装到数组中
        for binding_index in range(start_binding, end_binding):
            if engine.binding_is_input(binding_index):
                input_binding_idxs.append(binding_index)
            else:
                output_binding_idxs.append(binding_index)
        return input_binding_idxs, output_binding_idxs

    @staticmethod
    def get_output_tensors(
        context: trt.IExecutionContext,
        host_inputs: List[torch.Tensor],
        input_binding_idxs: List[int],
        output_binding_idxs: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        保留 GPU 内存, 简单来说就是用 torch.empty 构建输出变量
        """
        # explicitly set dynamic input shapes, so dynamic output shapes can be computed internally
        for host_input, binding_index in zip(host_inputs, input_binding_idxs):
            # Set the dynamic shape of a binding. 设置动态形状, 根据这个输入的形状
            context.set_binding_shape(binding_index, tuple(host_input.shape))
        # assert context.all_binding_shapes_specified
        device_outputs: Dict[str, torch.Tensor] = dict()
        for binding_index in output_binding_idxs:
            # 获取输出的形状
            # TensorRT computes output shape based on input shape provided above
            output_shape = context.get_binding_shape(binding=binding_index)
            # 输出的名字
            output_name = context.engine.get_binding_name(index=binding_index)
            # 分配 GPU 内存空间
            # allocate buffers to hold output results
            device_outputs[output_name] = torch.empty(tuple(output_shape), device="cuda")
        return device_outputs

    def infer_tensorrt(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        执行推理
        Perform inference with TensorRT.
        :param context: shared variable
        :param inputs: input tensor
        :param input_binding_idxs: input tensor indexes
        :param output_binding_idxs: output tensor indexes
        :return: output Dict[tensor name, tensor value]
        """
        # 打补丁的方式, 将需要的 self 属性都写在前面
        context = self.context
        get_output_tensors = self.get_output_tensors
        input_binding_idxs = self.input_binding_idxs
        output_binding_idxs = self.output_binding_idxs

        # 算了, 手动复制到 cuda 上吧
        inputs = dict((k, v.to("cuda")) for k, v in inputs.items())

        input_tensors: List[torch.Tensor] = list()
        # 对于每个名字的索引位置. 现在是按名字读取的输入, 所以 dict 的顺序就不重要了
        for i in range(context.engine.num_bindings):
            # 判断是否是输入位置
            if not context.engine.binding_is_input(index=i):
                continue
            # 输入的名字
            tensor_name = context.engine.get_binding_name(i)
            assert tensor_name in inputs, f"input not provided: {tensor_name}"
            tensor = inputs[tensor_name]
            assert isinstance(tensor, torch.Tensor), f"unexpected tensor class: {type(tensor)}"
            # 需要在 cuda 上. 原来如此, 旧的 v0.4.0 版本会手动给你复制到 cuda 上, 所以不会有这个报错
            assert (
                tensor.device.type == "cuda"
            ), f"unexpected device type (trt only works on CUDA): {tensor.device.type}"
            # 类型会被截断到 int32 上
            # warning: small changes in output if int64 is used instead of int32
            if tensor.dtype in [torch.int64, torch.long]:
                logging.warning(f"using {tensor.dtype} instead of int32 for {tensor_name}, will be casted to int32")
                tensor = tensor.type(torch.int32)
            input_tensors.append(tensor)
        # 继续深入, 实际上还是要看这个函数
        # calculate input shape, bind it, allocate GPU memory for the output
        outputs: Dict[str, torch.Tensor] = get_output_tensors(
            context, input_tensors, input_binding_idxs, output_binding_idxs
        )
        # data_prt 返回第一个元素的地址 Returns the address of the first element of self tensor.
        bindings = [int(i.data_ptr()) for i in input_tensors + list(outputs.values())]
        # Asynchronously execute inference on a batch. 这个是异步执行的, 所以下面需要使用 synchronize
        assert context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        ), "failure during execution of inference"
        # 等待完成, 相当于强制同步
        torch.cuda.current_stream().synchronize()  # sync all CUDA ops

        return outputs


if __name__ == "__main__":
    from transformers import BertTokenizerFast
    engine = load_engine("./onnx/model_torch.trt")
    infer = Infer(engine)
    enc = BertTokenizer.from_pretrained("bert-base-uncased")

    masked_sentences = [
        "Paris is the [MASK] of France.",
        "The primary [MASK] of the United States is English.",
        "A baseball game consists of at least nine [MASK].",
        "Topology is a branch of [MASK] concerned with the properties of geometric objects that remain unchanged under continuous transformations.",
    ]
    pos_masks = [4, 3, 9, 6]
    inputs = enc(masked_sentences[:1], return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    result = infer.infer_tensorrt(inputs)
    print(result)

