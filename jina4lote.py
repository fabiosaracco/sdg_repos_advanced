import os, sys, pickle
import numpy as np

from tqdm.auto import tqdm, trange

import tiktoken
from hf_token import token

from sentence_transformers import SentenceTransformer, util

# Carica il tokenizer per text-embedding-ada-002
enc = tiktoken.encoding_for_model("text-embedding-ada-002")

TOKENIZERS_PARALLELISM=True

# jina-embeddings-2
max_tokens=8192

model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, token=token)


def jina4lote(long_text, full_output=False):
    '''
    From the butchered text, embed each chunk and embed it. 
    Finally, take a weighted average, taking the leng of each chunk as weight.
    '''
    chunked_text=text_butcher(long_text)
    embeddings=[]
    for cut in tqdm(chunked_text, leave=False):
        embedding = model.encode(cut)
        embeddings.append(embedding)
    chunks_len=[len(chunk) for chunk in chunked_text]
    if full_output:
        return embeddings, np.average(embeddings, axis=0, weights=chunks_len)
    else:
        return np.average(embeddings, axis=0, weights=chunks_len)

def text_butcher(text, max_tokens=max_tokens):
    '''
    Cut the original text in pieces that can be used to feed Jina.
    The strategy is to subsequently cut the text any time it finds a period mark (.)
    compatible with the number of token handled by Jina.
    '''
    chunked_texts=[]
    new_text=text
    while len(new_text)>0:
        cacca=enc.encode(new_text)
        chunked_text=enc.decode(cacca[:max_tokens])
        if len(cacca)<max_tokens:
            # if cacca is smaller than max_tokens, we take everything...
            chunked_texts.append(chunked_text)
            new_text=''#new_text[len(chunked_text):].strip()
        elif '.' in chunked_text:
            # ... otherwise, cut the text up to the last sentence appropriately
            chunked_text='.'.join(chunked_text.split('.')[:-1])+'.'
            chunked_texts.append(chunked_text)
            new_text=new_text[len(chunked_text):].strip()
        else:
            raise Exception('WTF?!')
    return chunked_texts

