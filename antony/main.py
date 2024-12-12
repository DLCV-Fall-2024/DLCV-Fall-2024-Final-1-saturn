from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
from model import *
import random
import numpy as np

device = 'cuda'

def set_seed(seed=42):
    random.seed(seed)                 # Python's built-in random module
    np.random.seed(seed)              # Numpy
    torch.manual_seed(seed)           # PyTorch
    torch.cuda.manual_seed(seed)      # If using GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensures determinism on GPU (if using CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    # Customize how to process and batch data here if needed
    return batch

if __name__ == "__main__":
    split = 'train'
    training_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=False)
    output_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset"
    exp_name = "baseline.json"
    output_name = os.path.join(output_dir,exp_name)

    # Create a PyTorch DataLoader
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=4,  # Define batch size
        shuffle=True,   # Shuffle dataset
        collate_fn=collate_fn  # Optionally pass a collate function
    )
    # Load model
    model = MyModel(device)
    batch_size = 1
    set_seed(100)
    output_data = {}
    model.eval()
    # Iterate through the DataLoader

    for datas in tqdm(training_dataloader):
        id_names,images,conversations = [],[],[]
        for data in datas:
            id_names.append(data['id'])
            images.append(data['image'])
            conversations.append(data['conversations'][0])
        prompts = []
        for conversation in conversations:
            prompts.append(f"{conversation['value']}. ASSISTANT:")
        
        result = model(images,prompts)
        for idx,id_name in enumerate(id_names):
            output_data[id_name] = result[idx].split("ASSISTANT: ")[1]
    with open(output_name, "w") as file:
        json.dump(output_data, file, indent=4)
        
    # training_dataloader = 