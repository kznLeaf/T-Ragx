from collections.abc import Callable
from typing import Any

import pandas as pd
from comet import download_model, load_from_checkpoint

from t_ragx.utils.device import get_comet_predict_kwargs

from .LangDetectModel import FastTextLangDetectModel


class CometAggregationModel:
    """
    用于多模型选优
    """

    model: Any
    get_lang: Callable[[str], str | None]

    def __init__(
        self,
        model_id="Unbabel/wmt22-cometkiwi-da",
        get_lang_func: Callable[[str], str | None] | None = None,
    ):
        if get_lang_func is None:
            fast_text_lang_detect_model = FastTextLangDetectModel()
            # 调用 fasttext 检测输出语言是否和目标语言相同
            self.get_lang = fast_text_lang_detect_model.get_lang
        else:
            self.get_lang = get_lang_func

        model_path = download_model(model_id)
        self.model = load_from_checkpoint(model_path)

    def get_blind_score(self, out_text_list, source_text, target_lang_code="en"):
        """
        打分逻辑, 对每个候选译文，构造 {"src": 源文, "mt": 候选译文, "ref": ""}
        返回所有候选译文的分数
        """
        comet_data = [
            {"src": source_text[0], "mt": out_text_list[i], "ref": ""}
            for i in range(len(out_text_list))
        ]
        scores = self.model.predict(
            comet_data, batch_size=8, **get_comet_predict_kwargs()
        )
        out_score = scores.scores
        for i in range(len(out_score)):
            # 检测输出文本的语言，如果不是目标语言，分数直接设置为0
            if self.get_lang(out_text_list[i]) != target_lang_code:
                out_score[i] = 0
        return out_score

    def combine_preds(self, pred_dict, source_text, target_lang_code="en"):
        """
        pred_dict 形如：
        {
          0: ["译句1_模型A", "译句2_模型A", ...],
          1: ["译句1_模型B", "译句2_模型B", ...],
        }
        逐句选择 COMET 分数最高的句子
        """
        blind_results = {
            k: self.get_blind_score(
                pred_dict[k], source_text, target_lang_code=target_lang_code
            )
            for k in pred_dict
        }
        score_df = pd.DataFrame.from_dict(
            {k: blind_results[k] for k in blind_results}, orient="columns"
        )
        best_pred_key = score_df.apply(
            lambda row: row.index[row.argmax()], axis=1
        ).to_list()

        combined_pred = []
        for i in range(len(best_pred_key)):
            key = best_pred_key[i]
            combined_pred.append(pred_dict[key][i])

        return combined_pred
