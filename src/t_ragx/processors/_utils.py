import base64
import json
import os.path
import pathlib
import tempfile
import urllib.request
from hashlib import md5

import numpy as np
import requests

from ..utils.heuristic import clean_text


def serialize_str(s):
    return json.dumps(s, ensure_ascii=False)


def en_text_search(text, keyword):
    if len(keyword) > len(text):
        return False
    text = text.casefold()
    keyword = keyword.casefold()
    return bool(
        f" {keyword} " in text
        or text == keyword
        or len(keyword) - 1 > len(text)
        and text[: len(keyword) + 1] == keyword + " "
        or len(keyword) - 1 > len(text)
        and text[-len(keyword) + 1 :] == " " + keyword
    )


def merge_glossary_index(df):
    """
    merge the glossary records if the index is not unique
    otherwise df.to_dict("index") will throw error
    """

    dup_index = df[df.index.duplicated()].index
    for idx in dup_index:
        for c in df.columns:
            new_list = []
            for arr in df.loc[idx, c]:
                new_list += arr.tolist()
            df.loc[idx, c] = [np.array(list(set(new_list)))] * len(df.loc[idx, c])

    if len(dup_index) > 0:
        df = df[~df.index.duplicated(keep="first")]

    return df


# heuristic glossary retrieval
def get_glossary(text, glossary_dict, max_k=10, lang_code="en", source_lang="ja"):
    """
    glossary_dict: 术语表的内存词典，结构：
    {
        "スライム": {                      # key: 源语言词条（DataFrame 的 index）
            "en": array(["Slime", ...]), # value: 各语言列，cell 是 numpy.ndarray
            "zh": array(["史莱姆", ...]),
        },
        "リムル": {
            "en": array(["Rimuru"]),
            "zh": array(["利姆鲁"]),
        },
    }

    text 是一句待翻译文本

    """
    text = clean_text(text)
    out_dict = {}
    count = 0

    # 反向匹配，遍历整个词典的所有条目，看哪些术语出现在了词典中
    for entry in glossary_dict:
        if lang_code not in glossary_dict[entry]:
            continue
        if (entry in text and source_lang != "en") or (
            source_lang == "en" and en_text_search(text, entry)
        ):
            skip_flag = False
            # check for glossary word being a component of a longer glossary word
            for ek in out_dict:
                if entry.casefold() in ek.casefold():
                    skip_flag = True
                    break
            if skip_flag:
                continue

            out_dict[entry] = glossary_dict[entry][lang_code].tolist()
            count += 1
            if count >= max_k:
                break

    return out_dict


def get_http_file_id(url):
    response = requests.head(url)
    # use ETag if available
    if "ETag" in response.headers:
        return response.headers["ETag"].replace('"', "")

    # use encoded url path if ETag is not available
    return md5(base64.urlsafe_b64encode(url.encode())).hexdigest()


def file_cacher(file_path, tempfolder=None):
    """
    If the input file_path is a http url, cache the file (by ETag if possible) to local tempfolder

    Args:
        file_path:
        tempfolder:

    Returns:

    """
    if tempfolder is None:
        tempfolder = tempfile.gettempdir() + "/t_ragx"
        pathlib.Path(tempfolder).mkdir(parents=True, exist_ok=True)
    out_path = file_path
    if "http" in file_path:
        file_id = get_http_file_id(file_path)
        file_extension = pathlib.Path(file_path).suffix
        out_path = f"{tempfolder}/{file_id}{file_extension}"

        if not os.path.isfile(out_path):
            urllib.request.urlretrieve(file_path, out_path)

    return out_path
