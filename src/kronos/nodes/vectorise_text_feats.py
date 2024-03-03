import logging
import multiprocessing
from contextlib import contextmanager
from typing import Generator, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from spacy.language import Language

from kronos.data_interfaces.node_dfs_data_interface import (
    TX_NTYPES,
    WV_NTYPES,
    NodeAttrKey,
    NodeDF,
    NodeDFs,
)

logger = logging.getLogger(__name__)


def prep_emb_input(list_node_df: List[NodeDF]) -> List[str]:
    if len(list_node_df) < 1:
        raise ValueError("Input list of node dataframes cannot be empty")

    logger.info(
        "Extracting text arrays from node dataframes of following "
        f"node types:\n{[[node_df.ntype.value for node_df in list_node_df]]}"
    )

    list_text_series = [node_df.df[NodeAttrKey.text.value] for node_df in list_node_df]

    logger.info(
        "Distribution of text array sizes for node dataframes of each "
        f"node type:\n{[len(a_series) for a_series in list_text_series]}"
    )

    text = list(pd.concat(list_text_series, ignore_index=True).unique())

    logger.info(f"{len(text)} unique text strings are set to be embedeed")

    return text


#
# Average Word Vectors
#


def embed_with_avg_word_vec(
    text: List[str], spacy_pipeline: Language
) -> Iterable[Tuple[str, np.ndarray]]:
    logger.info(
        f"Embedding text array shaped of length {len(text)} "
        "with average word vectors"
    )

    n_process = max(1, multiprocessing.cpu_count() // 4)

    logger.info(f"Using {n_process} processes for spacy pipe operation")

    for doc in spacy_pipeline.pipe(texts=text, n_process=n_process):
        yield (doc.text, doc.vector)  # type: ignore[misc]


def _vectorise_with_word_vector(
    node_dfs: NodeDFs, spacy_pipeline: Language
) -> Iterable[Tuple[str, np.ndarray]]:
    text = prep_emb_input(
        list_node_df=[
            node_df for node_df in node_dfs.members if node_df.ntype in WV_NTYPES
        ]
    )
    return embed_with_avg_word_vec(text=text, spacy_pipeline=spacy_pipeline)


#
# Sentence Transformer Embeddings
#


@contextmanager
def multi_process_context(  # type: ignore[no-any-unimported]
    sent_tx: SentenceTransformer,
) -> Generator[dict, None, None]:
    pool = (
        sent_tx.start_multi_process_pool()
    )  # Default either with all CUDA or four CPUs

    try:  # Use try-finally block to make sure resources are stopped regardless
        yield pool
    finally:
        sent_tx.stop_multi_process_pool(pool)


def embed_with_sent_tx(  # type: ignore[no-any-unimported]
    text: List[str], sent_tx: SentenceTransformer, use_multi: bool = False
) -> Iterable[Tuple[str, np.ndarray]]:
    logger.info(
        f"Embedding text array of length {len(text)} with a sentence transformer"
    )

    # Use one process for small data
    if not use_multi:
        emb = sent_tx.encode(sentences=text, show_progress_bar=True)
        for i_text in range(len(text)):
            yield text[i_text], emb[i_text]

        return

    # Calculate the total number of batches
    batch_size = min(512, len(text))
    n_batch = len(text) // batch_size + (0 if len(text) % batch_size == 0 else 1)

    # Use context manager to prevent resource leak
    with multi_process_context(sent_tx=sent_tx) as pool:
        logger.info(
            "Using the following pool configuration for sentence "
            f"transformer multi encoding:\n{pool}"
        )

        # Batch encode input
        for i_batch in range(n_batch):
            # Calculate start and end index for the current batch
            start_idx = i_batch * batch_size
            end_idx = min(start_idx + batch_size, len(text))

            # Identify input chunk
            text_chunk = text[start_idx:end_idx]

            # Execute embedding calculation
            emb_chunk = sent_tx.encode_multi_process(
                sentences=text, pool=pool, batch_size=64
            )  # Double default batch size

            # Yield one pairs of text-embedding at a time
            for i_text in range(len(text_chunk)):
                yield text_chunk[i_text], emb_chunk[i_text]


def _vectorise_with_sentence_transformer(  # type: ignore[no-any-unimported]
    node_dfs: NodeDFs, sentence_transformer: SentenceTransformer
) -> Iterable[Tuple[str, np.ndarray]]:
    text = prep_emb_input(
        list_node_df=[
            node_df for node_df in node_dfs.members if node_df.ntype in TX_NTYPES
        ]
    )
    return embed_with_sent_tx(text=text, sent_tx=sentence_transformer)
