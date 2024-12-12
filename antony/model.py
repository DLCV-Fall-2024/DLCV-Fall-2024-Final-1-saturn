import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

class MyModel(nn.Module):
    def __init__(self,device):
        super(MyModel, self).__init__()
        # Define layers
        self.device = device
        model_id = "llava-hf/llava-1.5-7b-hf"
        self.model = LlavaForConditionalGeneration.from_pretrained(
                        model_id, 
                        torch_dtype=torch.float16, 
                        low_cpu_mem_usage=True,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def forward(self, images,prompts):
        # Define forward pass
        inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.device, torch.float32)
        generate_ids = self.model.generate(**inputs, max_new_tokens=500,num_beams=2,repetition_penalty=1.1 )
        result = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        return result