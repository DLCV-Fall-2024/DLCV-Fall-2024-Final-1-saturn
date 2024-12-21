import json
import os
from tqdm import tqdm
import numpy as np
import copy


if __name__ == "__main__":
    splits = ['train','val','test']
    input_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset"
    output_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset"
    filename = "@_v3.json"
    for split in splits:
        input_path = os.path.join(input_dir,filename.replace('@',split))
        with open(input_path, 'r') as file:
            datas = json.load(file)
        suggestion,general,regional = [],[],[]
        output_suggestion = os.path.join(output_dir,f'{split}_v3_suggestion.json')
        output_general = os.path.join(output_dir,f'{split}_v3_general.json')
        output_regional = os.path.join(output_dir,f'{split}_v3_regional.json')
        for data in datas:
            if('suggestion' in data['id']):
                suggestion.append(data)
            elif('general' in data['id']):
                general.append(data)
            else:
                regional.append(data)
        with open(output_suggestion, "w") as json_file:
            json.dump(suggestion, json_file, indent=4)
        with open(output_general, "w") as json_file:
            json.dump(general, json_file, indent=4)
        with open(output_regional, "w") as json_file:
            json.dump(regional, json_file, indent=4)

