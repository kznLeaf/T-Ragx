import json
import logging
from hashlib import sha1

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.notebook import tqdm

from ..models.constants import LANG_BY_LANG_CODE
from ..processors.constants import DEFAULT_MEMORY_INDEX
from .heuristic import clean_text, is_noise
from .heuristic import lang_detect as heuristic_lang_detect

logger = logging.getLogger("t_ragx")


def index_doc(df, index="translation_memory_demo"):
    """
    Formatted index action generator helper to help upload records to Elasticsearch

    Args:
        df:
        index:

    Returns:

    """
    for record in df.to_dict(orient="records"):
        # pop none
        for k in record:
            if record[k] is None:
                record.pop(k)
        yield (
            '{{ "index" : {{ "_index" : "{}", "_id": "{}"}}}}'.format(
                index, sha1(record[record["id_key"]].encode("utf8")).hexdigest()
            )
        )
        yield json.dumps(record, default=int)


def upsert_doc(df: pd.DataFrame, index: str | None = None):
    """
    Formatted upsert action generator helper to help upload records to Elasticsearch

    Args:
        df:
        index:

    写入后的文档类似:
        {
          "_id": "a1b2c3...",
          "_source": {
            "ja": "こんにちは",
            "en": "Hello",
            "id_key": "ja"
          }
        }
    """
    if index is None:
        index = DEFAULT_MEMORY_INDEX

    for record in df.to_dict(orient="records"):
        # pop none
        pop_list = []
        for k in record:
            if record[k] is None:
                pop_list.append(k)

        for k in pop_list:
            record.pop(k)
        yield (
            '{{ "update" : {{"_index" : "{}", "_id" : "{}", "retry_on_conflict" : 3}} }}'.format(
                index, sha1(record[record["id_key"]].encode("utf8")).hexdigest()
            )
        )
        # 有则更新，无则插入
        yield f'{{ "doc" : {json.dumps(record, default=int)}, "doc_as_upsert" : true }}'


def filter_df(df: pd.DataFrame, source_lang: str = "ja", lang_cols: list | None = None):
    """
    清洗写入 ES 之前的语句
    """
    # 确认要处理的语言列。为空则处理 en ja zh
    if lang_cols is None:
        lang_cols = list(LANG_BY_LANG_CODE.keys())  # en ja zh

    # 只保留实际存在的列
    lang_cols = list(set(lang_cols).intersection(df.columns))

    df.dropna(subset=lang_cols, how="all", inplace=True)
    df.drop_duplicates(subset=[source_lang], inplace=True)
    df[source_lang] = df[source_lang].apply(clean_text)
    # 去掉纯数字和日期
    df = df[~df[source_lang].map(is_noise)]
    df.reset_index(drop=True, inplace=True)

    # 去掉含有换行符的数据
    for c in lang_cols:
        df = df[~df[c].str.contains("\n", na=False)]

    # 按长度过滤
    for c in lang_cols:
        if c in ["ja", "zh"]:
            str_len = df[c].str.len()
            df = df[((350 > str_len) & (str_len > 4)) | (str_len.isna())]
        elif c in ["en"]:
            word_count = df[c].str.split(" ").str.len()
            df = df[((100 > word_count) & (word_count > 3)) | (word_count.isna())]

    # 统计每一列 日/英/中 字符数量，取最多的作为检测语言
    # 删掉「列名语言」和「检测到的语言」不一致的行。
    # 防止脏数据影响
    for c in lang_cols:
        # 检测这一列中的每个文本
        detected_lang = df[c].apply(heuristic_lang_detect)
        # 列名与该语言的检测语言一致，或者检测不到语言，都保留
        df = df[(c == detected_lang) | (detected_lang.isna())]

    df.reset_index(drop=True, inplace=True)

    return df


def upload_df(
    df: pd.DataFrame,
    es_client: Elasticsearch,
    id_key: str = "ja",
    batch_size: int = 10000,
    index: str | None = None,
) -> None:
    """
    upload_df 清洗语料 批量写入ES

    Args:
        df:
        es_client:
        id_key: The language column to hash (sha1) as ID. Duplicate records with common id will be merged.
                        id_key should be in df.columns 默认值ja, 用于 sha 取 index，去重
        batch_size:
        index: Defaulted to be "translation_memory". Should be explicitly set for in-task memories

    Returns:

    """
    df = filter_df(df, source_lang=id_key)
    # 新增一列
    df["id_key"] = id_key
    if len(df) < 1:
        print("Empty dataset")
        return
    batch_idx = np.array_split(range(len(df)), max(int(len(df) / batch_size), 1))
    for select_idx in tqdm(batch_idx):
        es_client.bulk(upsert_doc(df.iloc[select_idx]), index)


def csv_to_elastic(
    file_path,
    id_key="ja",
    elasticsearch_host: str = "localhost",
    es_client: Elasticsearch = None,
    batch_size=10000,
    read_csv_config: dict | None = None,
    index=None,
    elastic_client_args: dict | None = None,
):
    """
    Upload a CSV file to Elasticsearch
    The input csv should be parallel texts with the language code as their header
    For example:
        | ja  | en        | zh    |
        |-----|-----------|-------|
        | 例1 | example 1 | 範例1 |
        |     |           |       |
        |     |           |       |


    Args:

        file_path:
        id_key: The language column to hash (sha1) as ID. Duplicate records with common id will be merged.
                        id_key should be in df.columns
        elasticsearch_host:
        es_client:
        batch_size:
        read_csv_config:
        index: Defaulted to be "translation_memory". Should be explicitly set for in-task memories
        elastic_client_args:

    Returns:

    """

    if elastic_client_args is None:
        elastic_client_args = {}
    if read_csv_config is None:
        read_csv_config = {}
    if es_client is None:
        es_client = Elasticsearch(
            elasticsearch_host,  # Elasticsearch endpoint
            **elastic_client_args,
        )

    df = pd.read_csv(file_path, **read_csv_config)
    assert len(df.columns) > 1, "The CSV file has only one column"

    if len(set(df.columns).intersection(LANG_BY_LANG_CODE.keys())) < 2:
        logger.warning(f"The columns of the CSV are {df.columns}")

    upload_df(df, es_client, id_key=id_key, batch_size=batch_size, index=index)
