import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import os
import numpy as np
from tqdm import tqdm

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-6B-224px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-224px')
splits = ['train','val','test']
for split in splits:
    output_dir = f"/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/pool/{split}" 
    img_dir = f"/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/{split}"
    img_filenames = [ os.path.join(img_dir,i) for i in os.listdir(img_dir) ]
    for img_filename in tqdm(img_filenames):
        scene_name = os.path.basename(img_filename).split(".")[0]
        output_path = os.path.join(output_dir,scene_name)
        image = Image.open(img_filename).convert('RGB')
        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        outputs = model(pixel_values)
        outputs = outputs.pooler_output[0].to(torch.float32).detach().cpu().numpy()
        np.save(output_path, outputs)