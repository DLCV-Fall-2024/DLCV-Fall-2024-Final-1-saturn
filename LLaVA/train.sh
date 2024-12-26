# train "general" checkpoint
bash scripts/v1_5/finetune_task_lora.sh ../dataset/train/general.json ../dataset/train/general/ ../llava-v1.5-7b-task-lora-general/

# train "regional" checkpoint
bash scripts/v1_5/finetune_task_lora.sh ../dataset/train/regional.json ../dataset/train/regional/ ../llava-v1.5-7b-task-lora-regional/

# train "suggestion" checkpoint
bash scripts/v1_5/finetune_task_lora.sh ../dataset/train/suggestion.json ../dataset/train/suggestion/ ../llava-v1.5-7b-task-lora-suggestion/