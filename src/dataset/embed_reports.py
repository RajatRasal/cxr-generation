import argparse
import os
from typing import get_args, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .create_section_files import SECTIONED_OUTPUT_FILENAME


ENCODER_MODEL = Literal[
    # "UCSD-VA-health/RadBERT-RoBERTa-4m",
    # "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "microsoft/BiomedVLP-CXR-BERT-general",
]
SECTION_NAME = Literal["impression", "findings"]


def load_model(model_name: ENCODER_MODEL, gpu: bool = False, eval: bool = True) -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if gpu and torch.cuda.is_available():
        model = model.cuda()
    if eval:
        model = model.eval()
    return tokenizer, model


def embed_tokens(model: AutoModel, tokens):
    with torch.no_grad():
        return model(**tokens).pooler_output.cpu()


def batch_embedding(model: AutoModel, tokenizer: AutoTokenizer, strs: List[str], gpu: bool = False, batch_size: int = 32, max_length: int = 150) -> torch.Tensor:
    embeddings = []
    for batch in tqdm(DataLoader(strs, batch_size)):
        tokens = tokenizer(batch, max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        if gpu and torch.cuda.is_available():
            tokens = tokens.to("cuda")
        embedding = embed_tokens(model, tokens)
        embeddings.append(embedding)
    return torch.cat(embeddings)


def get_embedding_filename(section_name: SECTION_NAME, model_name: ENCODER_MODEL):
    model_name = model_name.replace("/", "__")
    return f'{section_name}_embeddings_{model_name}'


def write_embedding_file(arr: np.ndarray, sections_path: str, section_name: SECTION_NAME, model_name: ENCODER_MODEL):
    filename = get_embedding_filename(section_name, model_name)
    np.save(os.path.join(sections_path, filename), arr)


def read_embedding_file(section_path: str, section_name: SECTION_NAME, model_name: ENCODER_MODEL) -> np.ndarray:
    filename = get_embedding_filename(section_name, model_name) + ".npy"
    return np.load(os.path.join(section_path, filename))


def main(args):
    df = pd.read_csv(os.path.join(args.sections_path, SECTIONED_OUTPUT_FILENAME), index_col="study")

    impression_mask = df.impression.isna()
    findings_mask = df.findings.isna()
    df.loc[impression_mask, "impression"] = ""
    df.loc[findings_mask, "findings"] = ""
    impression_list = df.impression.tolist()
    findings_list = df.findings.tolist()

    for model_name in get_args(ENCODER_MODEL):
        tokenizer, model = load_model(model_name, args.gpu, eval=True)
        impression_embeddings = batch_embedding(model, tokenizer, impression_list, args.gpu, args.batch_size)
        write_embedding_file(impression_embeddings.numpy(), args.sections_path, "impression", model_name)
        findings_embeddings = batch_embedding(model, tokenizer, findings_list, args.gpu, args.batch_size)
        write_embedding_file(findings_embeddings.numpy(), args.sections_path, "findings", model_name)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections_path", type=str, required=True, help="Path to folder with CSV sections file")
    parser.add_argument("--gpu", action="store_true", help="If included, embeddings are computed on GPU")
    parser.add_argument("--batch_size", type=int, help="Batch size to use for embeddings model", default=32)
    parser.add_argument("--max_token_length", type=int, help="Max number of tokens to output from text encoder", default=150)

    args = parser.parse_args()
    main(args)
