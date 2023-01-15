"""
测试下 trt_infer.py 文件
"""

import torch
import numpy
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
    试下 fp16 TODO: fp16 下还是各种输出 nan
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


def print_result(result, masked_sentences, pos_masks):
    """
    打印出模型预测的结果, 也就是推断出的 [MASK] 位置上的词
    """
    most_likely_token_ids = [numpy.argmax(result[i, pos, :]) for i, pos in enumerate(pos_masks)]
    most_likely_probs = [numpy.max(result[i, pos, :]) for i, pos in enumerate(pos_masks)]
    print(most_likely_token_ids, most_likely_probs)
    unmasked_tokens = enc.decode(most_likely_token_ids).split(" ")
    unmasked_sentences = [masked_sentences[i].replace("[MASK]", token) for i, token in enumerate(unmasked_tokens)]
    for sentence in unmasked_sentences:
        print(sentence)


trt_fp16_model = ONNXTensorrtFp16Model()
e = trt_fp16_model.predict(masked_sentences[:1])[0]
print("e:", e)
exit()
# 准确性测试
print("===========开始准确性测试")
torch_model = TorchModel()
onnx_model = ONNXModel()
trt_model = ONNXTensorrtModel()
trt_compile_model = ONNXTensorrtCompileModel()
trt_fp16_model = ONNXTensorrtFp16Model()

a = torch_model.predict(masked_sentences[:1])[0]
b = onnx_model.predict(masked_sentences[:1])[0]
c = trt_model.predict(masked_sentences[:1])[0]
d = trt_compile_model.predict(masked_sentences[:1])[0]
e = trt_fp16_model.predict(masked_sentences[:1])[0]

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)
print("e:", e)

print("在 1e-3 级别上")
print("模型输出的一致性, a 和 b: ", numpy.allclose(a, b, rtol=1e-03, atol=1e-3))
print("模型输出的一致性, a 和 c: ", numpy.allclose(a, c, rtol=1e-03, atol=1e-3))
print("模型输出的一致性, a 和 d: ", numpy.allclose(a, d, rtol=1e-03, atol=1e-3))
print("模型输出的一致性, a 和 e: ", numpy.allclose(a, e, rtol=1e-03, atol=1e-3))

print("模型预测的值")
print_result(a, masked_sentences[:1], pos_masks[:1])
print_result(b, masked_sentences[:1], pos_masks[:1])
print_result(c, masked_sentences[:1], pos_masks[:1])
print_result(d, masked_sentences[:1], pos_masks[:1])
print_result(e, masked_sentences[:1], pos_masks[:1])

# 性能测试
print("===========开始性能测试")
