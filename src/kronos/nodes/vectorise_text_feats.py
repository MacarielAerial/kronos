from contextlib import contextmanager
import multiprocessing
from typing import Iterable, List, Tuple
import logging

import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from spacy.language import Language
from sentence_transformers import SentenceTransformer

from kronos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDFs

logger = logging.getLogger(__name__)

#
# Average Word Vectors
#

def embed_with_word_vectors(list_text_series: List[Series], spacy_pipeline: Language) -> Iterable[str, np.ndarray]:
    # TODO: Feed a list of strings directly and move database operation logic up the chain
    # Concatenate text series together and extract underlying numpy array
    text = pd.concat(list_text_series, ignore_index=True).to_numpy()

    logger.info(f"Embedding text array shaped {text.shape} with average word vectors")

    n_process = max(1, multiprocessing.cpu_count() // 4)

    logger.info(f"Using {n_process} processes for spacy pipe operation")

    for doc in spacy_pipeline.pipe(texts=text, n_process=n_process):
        yield (doc.text, doc.vector)

def _vectorise_with_word_vectors(node_dfs: List[NodeDFs]):
    # TODO: Add logic to parse text attributes from general node attribute dataframes as input for embedding

#
# Sentence Transformer Embeddings
#

@contextmanager
def multi_process_context(st: SentenceTransformer) -> Iterable[dict]:
    model = SentenceTransformer(st)
    pool = model.start_multi_process_pool()  # Default either with all CUDA or four CPUs

    try:  # Use try-finally block to make sure resources are stopped regardless
        yield pool
    finally:
        model.stop_multi_process_pool(pool)

def embed_with_st(list_text_series: List[Series], st: SentenceTransformer) -> Iterable[str, np.ndarray]:
    # Concatenate text series together and extract underlying numpy array
    text = pd.concat(list_text_series, ignore_index=True).to_list()

    logger.info(f"Embedding text array shaped {text.shape} with a sentence transformer")

    # Calculate the total number of batches
    batch_size = min(512, len(text))
    n_batch = len(text) // batch_size + (0 if len(text) % batch_size == 0 else 1)

    # Use context manager to prevent resource leak
    with multi_process_context(st=st) as pool:
        logger.info("Using the following pool configuration for sentence "
                    f"transformer multi encoding:\n{pool}")

        # Batch encode input
        for i_batch in range(n_batch):
            # Calculate start and end index for the current batch
            start_idx = i_batch * batch_size
            end_idx = start_idx + batch_size

            # Identify input chunk
            text_chunk = text[start_idx: end_idx]

            # Execute embedding calculation
            emb_chunk = st.encode_multi_process(sentences=text, pool=pool, 
                                batch_size=64)  # Double default batch size
            
            # Yield one pairs of text-embedding at a time
            for i_text in text_chunk:
                yield text_chunk[i_text], emb_chunk[i_text]
