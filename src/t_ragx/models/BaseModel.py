import abc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from t_ragx.utils.device import get_torch_device

from .constants import LANG_BY_LANG_CODE


def pretext_to_text(pretext_list, max_sent=5):
    """将文档前文列表格式化为 Prompt 片段。

    Args:
        pretext_list: 当前句之前的源语言句子列表；None 或空列表时不输出前文块。
        max_sent: 最多保留几句前文，默认 5。

    Example:
        >>> pretext_to_text(["你好。", "世界。"])
        'Preceding text:\\n  你好。\\n  世界。\\n\\n'
        >>> pretext_to_text(None)
        ''
    """
    if pretext_list is None or len(pretext_list) < 1:
        return ""

    out_text = "Preceding text:\n"
    for source_text in pretext_list[:max_sent]:
        out_text += f"  {source_text}\n"
    return out_text + "\n"


def glossary_to_text(glossary):
    """将术语表 dict 格式化为 Prompt 片段。

    Args:
        glossary: {"源词": ["译法1", "译法2"], ...}

    Example:
        >>> glossary_to_text({"スライム": ["史莱姆", "粘液"]})
        'Relevant Dictionary records:\\n  スライム: 史莱姆, 粘液\\n'
    """
    out_text = "Relevant Dictionary records:\n"
    for source_text in glossary:
        out_text += f"  {source_text}: {', '.join(glossary[source_text])}\n"
    return out_text


def trans_mem_to_text(trans_mem: list, source_lang_code="ja", target_lang_code="en"):
    """将 ES 检索到的翻译记忆格式化为 Prompt 片段。

    Args:
        trans_mem: 记忆列表，每项含源/目标语言字段，如 {"ja": "...", "en": "..."}。
        source_lang_code: 源语言列名，默认 "ja"。
        target_lang_code: 目标语言列名，默认 "en"。

    Example:
        >>> trans_mem_to_text([{"ja": "こんにちは", "en": "Hello"}], "ja", "en")
        'Examples translations:\\n 1. \\n   こんにちは\\n   Hello\\n'
        >>> trans_mem_to_text([])
        ''
    """
    if len(trans_mem) < 1:
        return ""
    out_text = "Examples translations:\n"
    count = 1
    for row in trans_mem:
        out_text += (
            f""" {count}. \n   {row[source_lang_code]}\n   {row[target_lang_code]}\n"""
        )
        count += 1
    return out_text


class BaseModel(metaclass=abc.ABCMeta):
    tokenizer = None
    model = None

    def __init__(self, model_id, adapter=None, tokenizer=None, model=None, device=None):
        """加载 HuggingFace 因果语言模型与分词器。

        Args:
            model_id: HuggingFace 模型 ID 或本地路径。
            adapter: 可选 LoRA adapter 路径（str 或 list[str]）。
            tokenizer: 可传入已加载的 tokenizer，为 None 时自动加载。
            model: 可传入已加载的 model，为 None 时自动加载。
            device: 设备名；None 时由 T_RAGX_DEVICE 或自动检测决定。

        Example:
            >>> model = MistralModel("rayliuca/TRagx-Mistral-7B-Instruct-v0.2")
            >>> model.tokenizer is not None and model.model is not None
            True
        """
        self.device = get_torch_device(device)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
                truncation_side="left",
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token

        if model is None:
            load_kwargs = {}
            if self.device.type == "cuda":
                load_kwargs["device_map"] = "auto"
            elif self.device.type == "mps":
                load_kwargs["torch_dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            if adapter is not None:
                if isinstance(adapter, list):
                    for a in adapter:
                        model.load_adapter(a)
                elif isinstance(adapter, str):
                    model.load_adapter(adapter)
                else:
                    ValueError(
                        "the adapter parameter must be either string or a list of strings"
                    )

            if self.device.type == "mps":
                model = model.to(self.device)

            model = model.eval()

        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, text_list=None, tokenize_config=None):
        """将 prompt 字符串列表编码为模型输入张量。

        Args:
            text_list: prompt 字符串列表。
            tokenize_config: 覆盖默认编码参数（max_length=2000, padding=True 等）。

        Example:
            >>> prompts = ["Translate: こんにちは"]
            >>> tokens = model.tokenize(prompts)
            >>> tokens["input_ids"].shape[0]  # batch size
            1
        """
        if text_list is None:
            text_list = []
        if tokenize_config is None:
            tokenize_config = {}

        default_tokenize_config = {
            "pad_to_multiple_of": 8,
            "padding": True,
            "truncation": True,
            "max_length": 2000,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        for k in default_tokenize_config:
            if k not in tokenize_config:
                tokenize_config[k] = default_tokenize_config[k]

        return self.tokenizer.batch_encode_plus(text_list, **tokenize_config).to(
            self.model.device
        )

    def generate(self, tokenized_input, generation_config=None):
        """调用底层 model.generate() 生成 token 序列。

        Args:
            tokenized_input: tokenize() 返回的 dict（input_ids, attention_mask 等）。
            generation_config: 生成参数，默认 max_new_tokens=100。

        Example:
            >>> tokens = model.tokenize(["Translate: hello"])
            >>> output_ids = model.generate(tokens)
            >>> output_ids.shape[0]  # 与输入 batch 大小一致
            1
        """
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 100,
                "early_stopping": True,
                "eos_token_id": [self.tokenizer.eos_token_id],
                "pad_token_id": self.tokenizer.eos_token_id,
            }

        for k in tokenized_input:
            tokenized_input[k] = tokenized_input[k].to(self.model.device)

        return self.model.generate(**tokenized_input, **generation_config)

    @staticmethod
    def clean_output(text):
        """清理模型原始输出（去除特殊 token 等）。子类必须实现。

        Example:
            >>> MistralModel.clean_output("Hello[/INST]")
            'Hello'
        """
        raise NotImplementedError

    def process_output(self, model_output, tokenized_input):
        """从完整生成序列中截取新增部分，解码并清理为译文列表。

        Args:
            model_output: generate() 返回的 token id 张量。
            tokenized_input: 原始输入 tokens，用于定位 prompt 长度。

        Example:
            >>> decoded = model.process_output(output_ids, token_data)
            >>> isinstance(decoded, list)
            True
        """
        translation_outputs = [
            o[len(i) :]
            for o, i in zip(
                model_output.cpu().numpy(), tokenized_input["input_ids"].cpu().numpy()
            )
        ]

        decoded_outputs = self.tokenizer.batch_decode(
            translation_outputs, skip_special_tokens=True
        )
        decoded_outputs = [self.clean_output(o) for o in decoded_outputs]
        return decoded_outputs

    def batch_translate(
        self,
        batch_text: list,
        source_lang_code="ja",
        target_lang_code="en",
        batch_search_result: list = None,
        batch_pre_text: list = None,
        tokenize_config=None,
        generation_config=None,
    ):
        """批量翻译：组装 prompt → 编码 → 生成 → 解码。

        Args:
            batch_text: 待译源句列表。
            batch_search_result: 与 batch_text 等长的检索结果，
                每项为 {"memory": [...], "glossary": {...}}。
            batch_pre_text: 与 batch_text 等长的前文列表，每项为 list[str] 或 None。

        Example:
            >>> results = model.batch_translate(
            ...     ["ラミリスの説明によると..."],
            ...     source_lang_code="ja",
            ...     target_lang_code="en",
            ...     batch_search_result=[{"memory": [], "glossary": {}}],
            ...     batch_pre_text=[["前の文です。"]],
            ... )
            >>> isinstance(results[0], str)
            True
        """
        query_prompts = self.batch_build_prompt(
            text=batch_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            pre_text_list=batch_pre_text,
            search_result=batch_search_result,
        )

        token_data = self.tokenize(query_prompts, tokenize_config)
        generation_output = self.generate(token_data, generation_config)
        translated_output = self.process_output(generation_output, token_data)
        return translated_output

    def translate(
        self,
        text: str,
        source_lang_code="ja",
        target_lang_code="en",
        search_result: list = None,
        pre_text: list = None,
        tokenize_config=None,
        generation_config=None,
    ):
        """单句翻译，内部将输入包装为长度为 1 的 batch 后调用 batch_translate()。

        Example:
            >>> model.translate(
            ...     "こんにちは",
            ...     search_result={"memory": [], "glossary": {}},
            ...     pre_text=None,
            ... )
            'Hello'  # 实际输出取决于模型
        """
        batch_text = [text]
        batch_pre_text = [pre_text]
        batch_search_result = [search_result]

        return self.batch_translate(
            batch_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            batch_search_result=batch_search_result,
            batch_pre_text=batch_pre_text,
            tokenize_config=tokenize_config,
            generation_config=generation_config,
        )[0]

    def batch_build_prompt(
        self,
        text: list,
        source_lang_code="Japanese",
        target_lang_code="English",
        search_result: list = None,
        pre_text_list: list = None,
    ):
        """为 batch 中每句分别构建 prompt，返回字符串列表。

        Args:
            text: 待译句列表。
            search_result: 与 text 等长；None 时视为无 RAG 上下文。
            pre_text_list: 与 text 等长；None 时视为无前文。

        Example:
            >>> prompts = model.batch_build_prompt(
            ...     ["句3"],
            ...     source_lang_code="ja",
            ...     target_lang_code="en",
            ...     search_result=[{"glossary": {}, "memory": []}],
            ...     pre_text_list=[["句1", "句2"]],
            ... )
            >>> "Preceding text:" in prompts[0]
            True
        """
        if pre_text_list is not None:
            assert len(pre_text_list) == len(text)
        else:
            pre_text_list = [None] * len(text)

        if search_result is not None:
            assert len(search_result) == len(text)
        else:
            search_result = [None] * len(text)

        return [
            self.build_prompt(
                t,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
                search_result=sr,
                pre_text=pt,
            )
            for t, sr, pt in zip(text, search_result, pre_text_list)
        ]

    def build_prompt(
        self,
        text,
        source_lang_code="ja",
        target_lang_code="en",
        search_result=None,
        pre_text: list = None,
    ):
        """为单句构建完整 chat prompt（术语 + 前文 + 记忆 + 翻译指令）。

        拼接顺序：glossary → pre_text → memory → 待译句。
        最终经 tokenizer.apply_chat_template() 转为模型所需格式。

        Example:
            >>> prompt = model.build_prompt(
            ...     "こんにちは",
            ...     source_lang_code="ja",
            ...     target_lang_code="en",
            ...     search_result={
            ...         "glossary": {"スライム": ["史莱姆"]},
            ...         "memory": [{"ja": "おはよう", "en": "Good morning"}],
            ...     },
            ...     pre_text=["前の文"],
            ... )
            >>> "Relevant Dictionary records:" in prompt
            True
            >>> "Preceding text:" in prompt
            True
            >>> "Examples translations:" in prompt
            True
            >>> "こんにちは" in prompt
            True
        """
        source_lang = LANG_BY_LANG_CODE[source_lang_code]
        target_lang = LANG_BY_LANG_CODE[target_lang_code]
        if search_result is None:
            search_result = {"glossary": [], "memory": []}

        chat = [
            {
                "role": "user",
                "content": (
                    "As a large language model, you are a trained expert in multiple languages. "
                    "These are some references that might help you translating passages:\n"
                    f"{glossary_to_text(search_result['glossary'])}{pretext_to_text(pre_text)}{trans_mem_to_text(search_result['memory'], source_lang_code=source_lang_code, target_lang_code=target_lang_code)}"
                    f"Translate this {source_lang} passage to {target_lang} "
                    "without additional questions, disclaimer, or explanations, but accurately and completely:"
                    f"{text}"
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
        )
