import os
import json

from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import shutil


PATH = "/vol/biodata/data/chest_xray/mimic-cxr-jpg/files/"
OUTPUT_PATH = "/vol/biomedic3/rrr2417/cxr-generation/notebooks/mimic_subset"

metadata = pd.read_csv("mimic-cxr-2.0.0-metadata.csv")
split = pd.read_csv("mimic-cxr-2.0.0-split.csv")
text = pd.read_csv("/vol/biomedic3/rrr2417/.cache/mimic_section/mimic_cxr_sectioned.csv")
text.study = text.study.str.slice(start=1)
text = text.astype({"study": "int32"})

df = pd.concat([metadata, split], axis=1)
df = df.loc[:,~df.columns.duplicated()].copy()

mask1 = df.ViewPosition == "PA"
mask2 = df.split == "train"

df_filter = df[mask1 & mask2]

metadata = []
for i, (_, row) in tqdm(enumerate(df_filter.iterrows()), total=df_filter.shape[0]):
    _row = text[text.study == row.study_id]
    impression = _row.impression
    if impression.size == 0:
        continue
    impression = impression.iloc[0]
    if str(impression) == 'nan':
        impression = ""
    dicom_id = row.dicom_id
    subject_id = row.subject_id
    study_id = row.study_id
    
    # flatten the file name and copy to folder
    # include file_name + impression in the jsonl
    old_path = os.path.join(PATH, f"p{str(subject_id)[:2]}", f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")
    metadata.append({"file_name": old_path, "text": impression})
    
with open(os.path.join(OUTPUT_PATH, "metadata.jsonl"), "w") as f:
    for m in metadata:
        f.write(json.dumps(m) + "\n")
