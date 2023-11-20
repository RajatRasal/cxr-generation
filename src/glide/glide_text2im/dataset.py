import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, get_args
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def extract_reports(reports_root_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # TODO: Verify reports zip SHA
    zip = os.path.join(reports_root_path, "mimic-cxr-reports.zip")
    with ZipFile(zip, "r") as f:
        f.extractall(out_dir)


@dataclass
class MIMICCXRReport:
    findings: Optional[str]
    impressions: Optional[str]


def parse_report(path: str) -> MIMICCXRReport:
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
    return MIMICCXRReport(findings, impression)


CXR_VIEWS = Literal["AP", "PA", "LATERAL", "LL"]


class MIMICCXR(Dataset):

    SPLIT_FILE = "mimic-cxr-2.0.0-split.csv.gz"
    METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"

    def __init__(self, images_root_path: str, split: Literal["train", "val", "test"], views: Optional[List[CXR_VIEWS]] = None, reports_extracted_path: Optional[str] = None):
        self.split = "validate" if split == "val" else split
        self.images_root_path = images_root_path
        self.reports_extracted_path = reports_extracted_path
        self.views = views

        # TODO: verify files and SHAs
        self.splits_file = pd.read_csv(os.path.join(self.images_root_path, self.SPLIT_FILE))
        self.metadata_file = pd.read_csv(os.path.join(self.images_root_path, self.METADATA_FILE))
        self.joined = pd.concat([self.metadata_file, self.splits_file], axis=1)
        self.joined = self.joined.loc[:, ~self.joined.columns.duplicated()]
        print(self.joined.ViewPosition.value_counts())
        print(self.joined.ViewPosition.isin(self.views).sum())
        mask = self.joined.split == self.split
        if self.views is not None:
            mask &= self.joined.ViewPosition.isin(self.views)
        self.df = self.joined[mask].reset_index()

        subject_id = self.df.subject_id.astype(str)
        study_id = self.df.study_id.astype(str)
        subpath = "/p" + subject_id.str[0:2] + "/p" + subject_id + "/s" + study_id
        images_path = "files" + subpath + "/" + self.df.dicom_id + ".jpg"
        self.images_path = images_path.map(lambda x: os.path.join(self.images_root_path, x))
        if self.reports_extracted_path is not None:
            reports_path = "files" + subpath + ".txt" 
            self.reports_path = reports_path.map(lambda x: os.path.join(self.reports_extracted_path, x))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        metadata = self.df.iloc[idx]
        image = Image.open(self.images_path.iloc[idx])
        report = None if self.reports_extracted_path is None else parse_report(self.reports_path.iloc[idx])
        return metadata, image, report


def main():
    reports_extracted_path = ".tmp/mimic_reports/"
    # extract_reports("/vol/biodata/data/chest_xray/mimic-cxr", reports_extracted_path)

    images_path = "/vol/biodata/data/chest_xray/mimic-cxr-jpg/"
    train = MIMICCXR(images_path, "train", views=["PA"])  # , reports_extracted_path)
    print(train.df.shape)

    for metadata, image, report in tqdm(train):
        break
        pass
        # print(np.array(image).shape)
        # break
