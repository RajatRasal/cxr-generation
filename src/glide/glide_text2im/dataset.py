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


def parse_report(path: str) -> Tuple[str, str]:
    # TODO: Make this more efficient - using a single pass
    findings = None
    impression = None
    with open(path, "r") as f:
        raw_report = f.read()
        raw_sections = re.sub(": \n \n ", ": ", raw_report).split("\n \n")
        for raw_section in raw_sections:
            if raw_section.startswith(" FINDINGS: "):
                findings = raw_section[len(" FINDINGS: "):].replace("\n", "")
            elif raw_section.startswith(" IMPRESSION: "):
                impression = raw_section[len(" IMPRESSION: "):].replace("\n", "")
    return findings, impression


CXR_VIEWS = Literal["AP", "PA"]


class CustomMIMICCXR(Dataset):

    def __init__(
        self,
        metadata_file_path: str,
        preproc_path: str,
        reports_path: Optional[str] = None,
        views: Optional[List[CXR_VIEWS]] = None
    ):
        self.metadata_file_path = metadata_file_path
        self.preproc_path = preproc_path
        self.reports_path = reports_path
        self.views = get_args(CXR_VIEWS) if views is None else views

        df = pd.read_csv(self.metadata_file_path)
        mask = df.ViewPosition.isin(self.views)
        self.df = df[mask].reset_index()
        subject_id = self.df.subject_id.astype(str)
        study_id = self.df.study_id.astype(str)
        reports_path_suffix = "files" + "/p" + subject_id.str[0:2] + "/p" + subject_id + "/s" + study_id + ".txt"
        self.full_reports_path = reports_path_suffix.map(lambda x: os.path.join(self.reports_path, x))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        metadata = self.df.iloc[idx]
        image_path = os.path.join(self.preproc_path, self.df.path_preproc.iloc[idx])
        image = Image.open(image_path)
        report = None if self.reports_path is None else parse_report(self.full_reports_path.iloc[idx])
        return metadata, image, report


def _batcher(x):
    return x


def format_reports():
    reports_extracted_path = "/vol/biomedic3/rrr2417/cxr-generation/.tmp/mimic_reports/"
    # reports_source = "/vol/biodata/data/chest_xray/mimic-cxr"
    # extract_reports("/vol/biodata/data/chest_xray/mimic-cxr", reports_extracted_path)
    report_data = []
    for i, (root, dirs, files) in enumerate(os.walk(reports_extracted_path)):
        # if i % 1000 == 0:
        if i == 2000:
            print(f"Parsed {i} reports...")
            break
        for file in files:
            if file.endswith(".txt"):
                report_file = os.path.join(root, file)
                report_data.append((report_file, *parse_report(report_file)))
    df = pd.DataFrame(report_data, columns=["full_report_path", "findings", "impression"])
    df.to_csv("reports.csv")


def main():
    metadata_file_path = "/vol/biomedic3/rrr2417/cxr-generation/metadata.csv"
    preproc_path = "/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/"
    reports_extracted_path = "/vol/biomedic3/rrr2417/cxr-generation/.tmp/mimic_reports/"

    dataset = CustomMIMICCXR(metadata_file_path, preproc_path, reports_extracted_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=15, collate_fn=_batcher)

    for i, x in enumerate(dataloader):
        if i % 10 == 0:
            print(i)

