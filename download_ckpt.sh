#!/bin/bash

# Google drive link: 'https://drive.google.com/drive/folders/1B0zK2pS6WClY2Q9_wR4W4DNF2t5u0MaS?usp=sharing'
gdown -O llava-v1.5-7b-task-lora-general.zip 'https://drive.google.com/uc?id=1DBu9kVcwA2uCD3Tq2w51CdiR7nmedM9G'
gdown -O llava-v1.5-7b-task-lora-regional.zip 'https://drive.google.com/uc?id=1ItBK2ItZCjkznFiBTpDNIcDeFpN6Rhx8'
gdown -O llava-v1.5-7b-task-lora-suggestion.zip 'https://drive.google.com/uc?id=1mMTczDtgTfNP8o_psInY_2U4XqIIwfXx'


unzip llava-v1.5-7b-task-lora-general.zip
unzip llava-v1.5-7b-task-lora-regional.zip
unzip llava-v1.5-7b-task-lora-suggestion.zip
rm llava-v1.5-7b-task-lora-general.zip
rm llava-v1.5-7b-task-lora-regional.zip
rm llava-v1.5-7b-task-lora-suggestion.zip
mv ./content/LLaVA/checkpoints/* ./
rm -fr ./content/LLaVA/checkpoints
