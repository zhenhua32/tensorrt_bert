"""
测试下 trt_infer.py 文件
"""

from functools import partial
import timeit

import torch
import numpy as np
import onnxruntime
from transformers import BertTokenizer, BertTokenizerFast, BertForMaskedLM

import trt_infer
from trt_infer import Input


# 全局变量
bert_model_path = "bert-base-uncased"
onnx_model_path = "./onnx/model.onnx"
trt_model_path = "./onnx/model_torch.trt"


# 准备下输入
enc = BertTokenizerFast.from_pretrained(bert_model_path)
masked_sentences = [
    "Paris is the [MASK] of France.",
    "The primary [MASK] of the United States is English.",
    "A baseball game consists of at least nine [MASK].",
    "Topology is a branch of [MASK] concerned with the properties of geometric objects that remain unchanged under continuous transformations.",
]
pos_masks = [4, 3, 9, 6]


class TorchModel:
    """
    原始 torch 模型, GPU 版本
    """

    def __init__(self, use_fast_tokenizer=True):
        if use_fast_tokenizer:
            self.enc = BertTokenizerFast.from_pretrained(bert_model_path)
        else:
            self.enc = BertTokenizer.from_pretrained(bert_model_path)
        self.model = BertForMaskedLM.from_pretrained(bert_model_path, torchscript=True).eval().cuda()

    def predict(self, texts: list, return_numpy=True):
        """
        texts 是数组
        pos_masks 是数组,  [MASK] 掩码所在的位置, 注意, 位置需要 +1, 因为前面会填充一个 [CLS] 字符
        'Paris is the [MASK] of France.' 对应的 pos_mask 是 4
        """
        inputs = self.enc(texts, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        inputs = dict((k, v.cuda()) for k, v in inputs.items())
        with torch.no_grad():
            result = self.model(**inputs)
        if return_numpy:
            result = [x.cpu().numpy() for x in result]
        return result


class ONNXModel:
    """
    ONNX 模型, GPU 版本
    """

    def __init__(self, use_fast_tokenizer=True):
        if use_fast_tokenizer:
            self.enc = BertTokenizerFast.from_pretrained(bert_model_path)
        else:
            self.enc = BertTokenizer.from_pretrained(bert_model_path)
        self.model = onnxruntime.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])

    def predict(self, texts: list):
        # 输入的数据类型是 np
        inputs = self.enc(texts, return_tensors="np", padding="max_length", max_length=128, truncation=True)
        result = self.model.run(None, input_feed=dict(inputs))
        return result


class ONNXTensorrtModel:
    """
    使用了预先编译的模型 ./onnx/model_torch.trt, 使用 polygraphy 导出的
    主要测试 trt_infer.load_engine

    # 将 onnx 导出成 tensort 格式
    !polygraphy convert ./onnx/model_torch.onnx --convert-to trt -o ./onnx/model_torch.trt --model-type onnx \
        --trt-min-shapes input_ids:[1,128] attention_mask:[1,128] token_type_ids:[1,128] \
        --trt-opt-shapes input_ids:[1,128] attention_mask:[1,128] token_type_ids:[1,128] \
        --trt-max-shapes input_ids:[1,128] attention_mask:[1,128] token_type_ids:[1,128]
    """

    def __init__(self, use_fast_tokenizer=True):
        if use_fast_tokenizer:
            self.enc = BertTokenizerFast.from_pretrained(bert_model_path)
        else:
            self.enc = BertTokenizer.from_pretrained(bert_model_path)

        engine = trt_infer.load_engine(trt_model_path)
        self.model = trt_infer.Infer(engine)

    def predict(self, texts: list, return_numpy=True):
        # 输入的数据类型是 pt
        inputs = self.enc(texts, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        # 将输入转换成 int32
        inputs = dict((k, v.to(torch.int32)) for k, v in inputs.items())
        # 推理, 和其他模型的输出保持一致, 实际上可以 ["logits"] 获取输出
        result = list(self.model.infer_tensorrt(inputs).values())
        if return_numpy:
            result = [x.cpu().numpy() for x in result]
        return result


class ONNXTensorrtCompileModel(ONNXTensorrtModel):
    """
    编译版, 主要测试 trt_infer.build_engine
    """

    def __init__(self, use_fast_tokenizer=True):
        if use_fast_tokenizer:
            self.enc = BertTokenizerFast.from_pretrained(bert_model_path)
        else:
            self.enc = BertTokenizer.from_pretrained(bert_model_path)

        input_shapes = [
            Input(
                "input_ids",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
            Input(
                "attention_mask",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
            Input(
                "token_type_ids",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
        ]

        engine = trt_infer.build_engine(
            onnx_model_path,
            inputs=input_shapes,
            max_workspace_size=1024 * 1024 * 1024 * 8,
        )
        self.model = trt_infer.Infer(engine)


class ONNXTensorrtFp16Model(ONNXTensorrtModel):
    """
    试下 fp16, 性能上也不是太明显
    """

    def __init__(self, use_fast_tokenizer=True):
        if use_fast_tokenizer:
            self.enc = BertTokenizerFast.from_pretrained(bert_model_path)
        else:
            self.enc = BertTokenizer.from_pretrained(bert_model_path)

        input_shapes = [
            Input(
                "input_ids",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
            Input(
                "attention_mask",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
            Input(
                "token_type_ids",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
        ]

        engine = trt_infer.build_engine(
            onnx_model_path,
            inputs=input_shapes,
            max_workspace_size=1024 * 1024 * 1024 * 8,
            fp16=True,
        )
        self.model = trt_infer.Infer(engine)


class ONNXTensorrtNotTf32Model(ONNXTensorrtModel):
    """
    禁用 tf32
    """

    def __init__(self, use_fast_tokenizer=True):
        if use_fast_tokenizer:
            self.enc = BertTokenizerFast.from_pretrained(bert_model_path)
        else:
            self.enc = BertTokenizer.from_pretrained(bert_model_path)

        input_shapes = [
            Input(
                "input_ids",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
            Input(
                "attention_mask",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
            Input(
                "token_type_ids",
                min_shape=(1, 128),
                opt_shape=(1, 128),
                max_shape=(1, 128),
            ),
        ]

        engine = trt_infer.build_engine(
            onnx_model_path,
            inputs=input_shapes,
            max_workspace_size=1024 * 1024 * 1024 * 8,
            tf32=False,
        )
        self.model = trt_infer.Infer(engine)


def print_result(model_name, result, masked_sentences, pos_masks):
    """
    打印出模型预测的结果, 也就是推断出的 [MASK] 位置上的词
    """
    most_likely_token_ids = [np.argmax(result[i, pos, :]) for i, pos in enumerate(pos_masks)]
    most_likely_probs = [np.max(result[i, pos, :]) for i, pos in enumerate(pos_masks)]
    print(model_name, most_likely_token_ids, most_likely_probs)
    unmasked_tokens = enc.decode(most_likely_token_ids).split(" ")
    unmasked_sentences = [masked_sentences[i].replace("[MASK]", token) for i, token in enumerate(unmasked_tokens)]
    for sentence in unmasked_sentences:
        print(sentence)


def time_graph(call_func, num_loops=100):
    """
    多次调用后, 返回每次调用时间的数组
    """
    print("预热 ...")
    for _ in range(20):
        call_func()

    # 等待同步, cuda 默认是异步调用的
    torch.cuda.synchronize()

    print("开始计时 ...")
    timings = []
    for i in range(num_loops):
        start_time = timeit.default_timer()
        call_func()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
        # print("Iteration {}: {:.6f} s".format(i, end_time - start_time))

    return timings


def print_stats(model_name, timings, batch_size):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    # 都用 ms 作为单位
    time_mean = np.mean(times) * 1000
    time_med = np.median(times) * 1000
    time_99th = np.percentile(times, 99) * 1000
    time_std = np.std(times, ddof=0) * 1000

    msg = f"""
    {model_name} =================================
    batch size={batch_size}, 迭代次数={steps}
    中位数 text batches/second: {speed_med}, 均值: {speed_mean}
    中位数 latency: {time_med:.2f}, 均值: {time_mean:.2f}, 99th_p: {time_99th:2f}, std_dev: {time_std:.2f}
    """
    print(msg)


# trt_fp16_model = ONNXTensorrtFp16Model()
# e = trt_fp16_model.predict(masked_sentences[:1])[0]
# print("e:", e)
# exit()
# 准确性测试
print("===========开始准确性测试")
torch_model = TorchModel()
onnx_model = ONNXModel()
trt_model = ONNXTensorrtModel()
trt_compile_model = ONNXTensorrtCompileModel()
trt_fp16_model = ONNXTensorrtFp16Model()
trt_not_tf32_model = ONNXTensorrtNotTf32Model()

a = torch_model.predict(masked_sentences[:1])[0]
b = onnx_model.predict(masked_sentences[:1])[0]
c = trt_model.predict(masked_sentences[:1])[0]
d = trt_compile_model.predict(masked_sentences[:1])[0]
e = trt_fp16_model.predict(masked_sentences[:1])[0]
f = trt_not_tf32_model.predict(masked_sentences[:1])[0]

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)
print("e:", e)
print("f:", f)

print("在 1e-3 级别上")
print("模型输出的一致性, a 和 b: ", np.allclose(a, b, rtol=1e-3, atol=1e-3))
print("模型输出的一致性, a 和 c: ", np.allclose(a, c, rtol=1e-3, atol=1e-3))
print("模型输出的一致性, a 和 d: ", np.allclose(a, d, rtol=1e-3, atol=1e-3))
print("模型输出的一致性, a 和 e: ", np.allclose(a, e, rtol=1e-3, atol=1e-3))
print("模型输出的一致性, a 和 f: ", np.allclose(a, f, rtol=1e-3, atol=1e-3))

print("模型预测的值")
print_result("a", a, masked_sentences[:1], pos_masks[:1])
print_result("b", b, masked_sentences[:1], pos_masks[:1])
print_result("c", c, masked_sentences[:1], pos_masks[:1])
print_result("d", d, masked_sentences[:1], pos_masks[:1])
print_result("e", e, masked_sentences[:1], pos_masks[:1])
print_result("f", f, masked_sentences[:1], pos_masks[:1])

# 性能测试
print("===========开始性能测试")
timings = time_graph(partial(torch_model.predict, masked_sentences[:1], False))
print_stats("BERT Torch GPU", timings, 1)

timings = time_graph(partial(onnx_model.predict, masked_sentences[:1]))
print_stats("BERT ONNX GPU", timings, 1)

timings = time_graph(partial(trt_model.predict, masked_sentences[:1], False))
print_stats("BERT ONNX TRT 预编译 GPU", timings, 1)

timings = time_graph(partial(trt_compile_model.predict, masked_sentences[:1], False))
print_stats("BERT ONNX TRT 运行时编译 GPU", timings, 1)

timings = time_graph(partial(trt_fp16_model.predict, masked_sentences[:1], False))
print_stats("BERT ONNX TRT FP16 GPU", timings, 1)

timings = time_graph(partial(trt_not_tf32_model.predict, masked_sentences[:1], False))
print_stats("BERT ONNX TRT 不使用TF32 GPU", timings, 1)
