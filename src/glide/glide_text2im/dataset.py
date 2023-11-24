import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, get_args
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


@dataclass
class MIMICCXRReport:
    findings: Optional[str]
    impressions: Optional[str]


CXR_VIEWS = Literal["AP", "PA"]


class CustomMIMICCXR(Dataset):

    def __init__(
        self,
        metadata_file_path: str,
        preproc_path: str,
        section_files_path: str,
        views: Optional[List[CXR_VIEWS]] = None
    ):
        self.metadata_file_path = metadata_file_path
        self.preproc_path = preproc_path
        self.section_files_path = section_files_path
        self.views = get_args(CXR_VIEWS) if views is None else views

        merged_section_file_path = os.path.join(self.section_files_path, "mimic_cxr_sectioned.csv")
        reports_df = pd.read_csv(merged_section_file_path)

        df = pd.read_csv(self.metadata_file_path)
        mask = df.ViewPosition.isin(self.views)
        df = df[mask].reset_index()
        df.study_id = "s" + df.study_id.astype(str)

        df = pd.merge(df, reports_df, how="left", left_on="study_id", right_on="study")
        # mask = ~(df.impression.isna() | df.findings.isna())
        self.df = df[mask]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.preproc_path, self.df.path_preproc.iloc[idx])
        image = Image.open(image_path)
        return row, image, row.findings, row.impression


def _batcher(x):
    return x


def main():
    metadata_file_path = "/vol/biomedic3/rrr2417/cxr-generation/metadata.csv"
    preproc_path = "/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/"
    reports_extracted_path = "/vol/biomedic3/rrr2417/cxr-generation/.tmp/mimic_section_files/"

    dataset = CustomMIMICCXR(metadata_file_path, preproc_path, reports_extracted_path)
    for x in dataset:
        print(x)
        break
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=15, collate_fn=_batcher)

    # for i, x in enumerate(dataloader):
    #     if i % 10 == 0:
    #         print(i)
