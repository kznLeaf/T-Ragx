import logging
from operator import itemgetter

from elasticsearch import Elasticsearch
from Levenshtein import distance
from tqdm.auto import tqdm

from ..utils.heuristic import clean_text
from .BaseInputProcessor import BaseInputProcessor

logger = logging.getLogger("t_ragx")


def _build_translation_memory_query(
    search_term: str,
    source_lang: str,
    target_lang: str,
    top_k: int,
    task_index: str | None,
    task_boost: float,
) -> dict:
    """Build the Elasticsearch query body for translation memory retrieval."""
    indices_boost = []
    if task_index is not None:
        indices_boost.append({task_index: task_boost})

    return {
        "size": top_k,
        "indices_boost": indices_boost,
        "_source": {"includes": [source_lang, target_lang, "source"]},
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": search_term,
                            "fields": [source_lang],
                            "escape": True,
                        }
                    },
                ],
                "filter": [
                    {"exists": {"field": target_lang}},
                ],
            }
        },
    }


def _elastic_hits(elastic_result) -> list:
    """Return hit list from an ES response, or an empty list for invalid responses."""
    if not isinstance(elastic_result, dict):
        return []
    return elastic_result.get("hits", {}).get("hits", [])


def rerank_elastic_result(elastic_result, source_lang, search_term, top_k=5):
    """Rerank BM25 hits by Levenshtein distance on the source-language field."""
    hits = _elastic_hits(elastic_result)
    if not hits:
        return []

    top_score = None
    result_list = []
    for hit in hits:
        if top_score is None:
            top_score = hit["_score"]
        if hit["_score"] < top_score and len(result_list) > top_k:
            break

        if source_lang not in hit["_source"]:
            continue
        hit["distance"] = distance(hit["_source"][source_lang], search_term)
        result_list.append(hit)

    result_list = sorted(result_list, key=itemgetter("distance"))
    return result_list[:top_k]


def search_single_elastic(
    es_client,
    index,
    search_term,
    source_lang,
    target_lang,
    top_k=10,
    request_timeout=50,
    task_index=None,
    task_boost=1.2,
):
    """Execute a single Elasticsearch translation memory query."""
    index_list = [index]
    if task_index is not None:
        index_list.append(task_index)

    return es_client.search(
        index=index_list,
        body=_build_translation_memory_query(
            search_term,
            source_lang,
            target_lang,
            top_k,
            task_index,
            task_boost,
        ),
        request_timeout=request_timeout,
    )


def search_elastic_with_retry(
    es_client,
    index,
    search_term,
    source_lang,
    target_lang,
    top_k=10,
    retry=3,
    task_index=None,
    task_boost=1.2,
):
    """Retry Elasticsearch search and return an empty hit structure on failure."""
    for attempt in range(retry):
        try:
            return search_single_elastic(
                es_client,
                index,
                search_term,
                source_lang,
                target_lang,
                top_k=top_k,
                task_index=task_index,
                task_boost=task_boost,
            )
        except Exception:
            logger.debug(
                "Elasticsearch search failed on attempt %s/%s",
                attempt + 1,
                retry,
                exc_info=True,
            )

    logger.warning("elastic time out")
    return {"hits": {"hits": []}}


def _truncate_hit_sources(hit, source_lang, target_lang, max_item_len):
    """Truncate source/target fields in a hit when max_item_len is positive."""
    if max_item_len <= 0:
        return
    source = hit["_source"]
    source[source_lang] = source[source_lang][:max_item_len]
    source[target_lang] = source[target_lang][:max_item_len]


def batch_search_elastic(
    es_client,
    index,
    search_term_list,
    source_lang,
    target_lang,
    top_k=10,
    rerank_top_k=5,
    pbar=False,
    task_index=None,
    task_boost=1.2,
    max_item_len=-1,
):
    """Search Elasticsearch for each query string and rerank the combined results."""
    bulk_result = []
    for search_term in tqdm(search_term_list, disable=(not pbar)):
        search_result = search_elastic_with_retry(
            es_client,
            index,
            search_term,
            source_lang,
            target_lang,
            top_k=top_k,
            task_index=task_index,
            task_boost=task_boost,
        )

        for hit in _elastic_hits(search_result):
            _truncate_hit_sources(hit, source_lang, target_lang, max_item_len)

        bulk_result.append(search_result)

    return [
        rerank_elastic_result(resp, source_lang, search_term, top_k=rerank_top_k)
        for search_term, resp in zip(search_term_list, bulk_result)
    ]


def _format_memory_hits(hits):
    """Convert reranked Elasticsearch hits into prompt-ready memory dicts."""
    return [
        {"score": hit["_score"], "distance": hit["distance"]} | hit["_source"]
        for hit in hits
    ]


def _add_normed_distance(query_text, memory_hits):
    """Add normalized Levenshtein distance relative to the query length."""
    query_len = len(query_text)
    if query_len == 0:
        return
    for hit in memory_hits:
        hit["normed_distance"] = hit["distance"] / query_len


class ElasticInputProcessor(BaseInputProcessor):
    """Default input processor backed by pre-built Elasticsearch translation memory indexes.

    Glossary loading and search are inherited from BaseInputProcessor.
    """

    def __init__(self, device=None):
        super().__init__(device=device)

    def load_general_translation(
        self,
        elastic_index="translation_memory",
        elasticsearch_host: str | list[str] = "localhost",
        es_client: Elasticsearch | None = None,
        elastic_client_args=None,
    ):
        """Connect to an existing Elasticsearch translation memory index.

        This override does not create or populate indexes. Call upload_df/csv_to_elastic
        separately when building a local index.
        """
        elastic_client_args = elastic_client_args or {}

        if es_client is None:
            es_client = Elasticsearch(elasticsearch_host, **elastic_client_args)

        if not es_client.indices.exists(index=elastic_index):
            raise ValueError(f"Elasticsearch index does not exist: {elastic_index}")

        self.es_client = es_client
        self.general_memory_elastic_index = elastic_index

    def search_general_memory(self, *args, **kwargs):
        """Alias for search_memory to match the BaseInputProcessor API."""
        return self.search_memory(*args, **kwargs)

    def search_memory(
        self,
        text_list: list[str] | str,
        search_index: str | None = None,
        source_lang="ja",
        target_lang="en",
        top_k=10,
        rerank_top_k=None,
        max_item_len=500,
        pbar=False,
        task_index=None,
        task_boost=1.2,
    ):
        """Search translation memory via Elasticsearch and return structured hit metadata."""
        if isinstance(text_list, str):
            text_list = [text_list]

        text_list = [clean_text(text) for text in text_list]
        if search_index is None:
            search_index = self.general_memory_elastic_index

        if rerank_top_k is None:
            rerank_top_k = top_k

        search_result_list = batch_search_elastic(
            self.es_client,
            search_index,
            text_list,
            source_lang,
            target_lang,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            pbar=pbar,
            task_index=task_index,
            task_boost=task_boost,
            max_item_len=max_item_len,
        )

        processed_output = [
            _format_memory_hits(search_result) for search_result in search_result_list
        ]

        for query_text, memory_hits in zip(text_list, processed_output):
            _add_normed_distance(query_text, memory_hits)

        return processed_output
