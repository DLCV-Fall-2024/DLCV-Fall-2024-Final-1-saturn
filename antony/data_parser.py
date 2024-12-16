from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
from model import *
import random
import numpy as np
import sys
import json
import cv2
from PIL import Image

def collate_fn(batch):
    # Customize how to process and batch data here if needed
    return batch

if __name__ == "__main__":
    output_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset"
    splits = ["train", "val", "test"]
    for split in splits:
        output_json_name = f"{split}.json"
        output_img_dir = os.path.join(output_dir,split)
        output_json_path = os.path.join(output_dir,output_json_name)
        os.makedirs(output_img_dir, exist_ok=True)
        # Load dataset
        output_json = []
        dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=False)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Define batch size
            shuffle=True,   # Shuffle dataset
            collate_fn=collate_fn  # Optionally pass a collate function
        )
        '''
                [
        {
            "id": "997bb945-628d-4724-b370-b84de974a19f",
            "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
            "conversations": [
            {
                "from": "human",
                "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
            },
            {
                "from": "gpt",
                "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
            },
            ]
        },
        ...
        ]
        '''
        for datas in tqdm(dataloader):
            batch_data = {}
            id = datas[0]['id']
            image = datas[0]['image']
            img_name = os.path.join(output_img_dir,f"{id}.png")
            batch_data['id'] = id
            batch_data['image'] = img_name
            batch_data['conversations'] = datas[0]['conversations']
            image.save(img_name)
            output_json.append(batch_data)
        with open(output_json_path, "w") as json_file:
            json.dump(output_json, json_file, indent=4)


