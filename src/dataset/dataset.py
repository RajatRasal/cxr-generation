import os
from dataclasses import dataclass
from functools import partial
from typing import List, Literal, Optional, TypedDict, get_args
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.embed_reports import ENCODER_MODEL, SECTION_NAME, read_embedding_file


def extract_reports(reports_root_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # TODO: Verify reports zip SHA
    _zip = os.path.join(reports_root_path, "mimic-cxr-reports.zip")
    with ZipFile(_zip, "r") as f:
        f.extractall(out_dir)


def get_metadata(splits_path: str, out_path: str):
    # TODO: Verify SHA
    train_df = pd.read_csv(os.path.join(splits_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(splits_path, "test.csv"))
    df = pd.concat([train_df, val_df, test_df])
    df.to_csv(os.path.join(out_path, "metadata.csv"))


class MIMICCXRData(TypedDict):
    image: torch.FloatTensor
    findings: str
    impression: str
    findings_embedding: torch.FloatTensor
    impression_embedding: torch.FloatTensor


class MIMICCXRBatch(TypedDict):
    image_batch: torch.FloatTensor
    findings_batch: List[str]
    impression_batch: List[str]
    findings_embedding_batch: torch.FloatTensor
    impression_embedding_batch: torch.FloatTensor


CXR_VIEW = Literal["AP", "PA"]


class CustomMIMICCXR(Dataset):

    def __init__(
        self,
        metadata_file_path: str,
        preproc_path: str,
        section_files_path: str,
        encoder: ENCODER_MODEL = "microsoft/BiomedVLP-CXR-BERT-general",
        views: Optional[List[CXR_VIEW]] = None,
        sections: Optional[List[SECTION_NAME]] = None,
    ):
        self.metadata_file_path = metadata_file_path
        self.preproc_path = preproc_path
        self.section_files_path = section_files_path
        self.views = get_args(CXR_VIEW) if views is None else views
        self.sections = get_args(SECTION_NAME) if sections is None else sections

        merged_section_file_path = os.path.join(self.section_files_path, "mimic_cxr_sectioned.csv")
        reports_df = pd.read_csv(merged_section_file_path)
        self.df_to_embed_map = {study_id: i for i, study_id in enumerate(reports_df.study.tolist())}

        if "findings" in self.sections or self.sections is None:
            # TODO: SHA CHECK
            self.findings_embeddings = read_embedding_file(self.section_files_path, "findings", encoder)
        if "impression" in self.sections or self.sections is None:
            # TODO: SHA CHECK
            self.impression_embeddings = read_embedding_file(self.section_files_path, "impression", encoder)

        metadata_df = pd.read_csv(self.metadata_file_path)
        metadata_df.study_id = "s" + metadata_df.study_id.astype(str)

        self.df = pd.merge(metadata_df, reports_df, how="inner", left_on="study_id", right_on="study")

        mask = self.df.ViewPosition.isin(self.views)
        self.df = self.df[mask]
        # mask = ~(df.impression.isna() | df.findings.isna())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> MIMICCXRData:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.preproc_path, self.df.path_preproc.iloc[idx])
        image = read_image(image_path)
        embed_idx = self.df_to_embed_map[row.study_id]
        return {
            "image": image,
            "findings": row.findings,
            "impression": row.impression,
            "findings_embedding": torch.tensor(self.impression_embeddings[embed_idx]).unsqueeze(0),
            "impression_embedding": torch.tensor(self.findings_embeddings[embed_idx]).unsqueeze(0),
        }


def mimic_cxr_batcher(items: List[MIMICCXRData]) -> MIMICCXRBatch:
    return {
        "image_batch": torch.cat([item["image"] for item in items]),
        "findings_batch": [item["findings"] for item in items],
        "impression_batch": [item["impression"] for item in items],
        "findings_embedding_batch": torch.cat([item["findings_embedding"] for item in items]),
        "impression_embedding_batch": torch.cat([item["impression_embedding"] for item in items]),
    }


CustomMIMICCXRDataLoader = partial(DataLoader, collate_fn=mimic_cxr_batcher)


def main():
    metadata_file_path = "/vol/biomedic3/rrr2417/cxr-generation/metadata.csv"
    preproc_path = "/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/"
    reports_extracted_path = "/vol/biomedic3/rrr2417/cxr-generation/.tmp/mimic_section_files/"

    dataset = CustomMIMICCXR(
        metadata_file_path,
        preproc_path,
        reports_extracted_path,
        views=["AP"],
        sections=["impression", "findings"],
    )

    dataloader = CustomMIMICCXRDataLoader(dataset, batch_size=3, shuffle=True, num_workers=15)
    for x in dataloader:
        print(x["image_batch"].shape)
        print(len(x["findings_batch"]))
        print(len(x["impression_batch"]))
        print(x["findings_embedding_batch"].shape)
        print(x["impression_embedding_batch"].shape)
        break

