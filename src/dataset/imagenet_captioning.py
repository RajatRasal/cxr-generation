import argparse
import os

import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flexit_csv", type=str, default="src/dataset/queries.csv")
    parser.add_argument("--imagenet_folder", type=str, default="/vol/biodata/data/ILSVRC2012/val/")
    parser.add_argument("--output_folder", type=str, default="src/dataset/")
    parser.add_argument("--model", type=str, required=True, choices=["blip2-opt-2.7b", "blip2-flan-t5-xxl", "blip2-opt-6.7b-coco", "blip2-opt-6.7b", "blip2-flan-t5-xl"])
    args = parser.parse_args()
    return args


def main():
    args = arguments()

    df = pd.read_csv(args.flexit_csv)

    model_name = "Salesforce/" + args.model

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, load_in_8bit=True, torch_dtype=torch.bfloat16
    )

    def qa(image, question, max_length=5): 
        inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
        generated_ids = model.generate(**inputs, max_length=max_length)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    caption_infos = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        image = Image.open(os.path.join(args.imagenet_folder, row.path))
        source = row.source.lower()
        location = qa(image, f"Question: Where is the {source}? Answer:", 5).lower()
        action = qa(image, f"Question: What is the {source} doing? Answer:", 1).lower()
        caption_infos.append((location, action))
    df_captions = pd.DataFrame(caption_infos, columns=["location", "action"])

    output_csv = os.path.join(args.output_folder, "caption_info_" + args.model.replace("-", "_").replace(".", "_")) + ".csv"
    df_captions.to_csv(output_csv)

