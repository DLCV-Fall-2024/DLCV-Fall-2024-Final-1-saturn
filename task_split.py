import os
import json
import argparse

"""
divide images in "train/" into 3 subfolders, "train/general/", "train/regional/" and "train/suggestion/"
train.json is also splitted into general.json, regional.json and suggestion.json

If --add_val is True (merge val data to train data), images in "val/" are migrated to the 3 subfolders based on question types.
Samples in val.json are also added to the 3 json files (general.json, regional.json and suggestion.json) in "train/"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="dataset/train/")
    parser.add_argument("--json_path", type=str, default="dataset/train.json")
    parser.add_argument("--add_val", type=bool, default=True)
    parser.add_argument("--extra_image_folder", type=str, default="dataset/val/")
    parser.add_argument("--extra_json_path", type=str, default="dataset/val.json")

    args = parser.parse_args()

    task_types = ["general", "regional", "suggestion"]
    lst_general, lst_regional, lst_suggestion = [], [], []
    for task in task_types:
        os.makedirs(os.path.join(args.image_folder, task), exist_ok=True)        

    with open(args.json_path, "r") as f:
        data_lst = json.load(f)

    for data in data_lst:
        data_type = data["id"].split("_")[1]
        new_image_path = os.path.join(args.image_folder, data_type, data["id"]+".png")
        # move image file
        os.rename(data["image"], new_image_path)
        # modify image path in data
        data["image"] = new_image_path
        # add to new jsons
        if data_type == "general":
            lst_general.append(data)
        elif data_type == "regional":
            lst_regional.append(data)
        elif data_type == "suggestion":
            lst_suggestion.append(data)

    if args.add_val:
        with open(args.extra_json_path, "r") as f:
            data_lst = json.load(f)
        
        for data in data_lst:
            data_type = data["id"].split("_")[1]
            new_image_path = os.path.join(args.image_folder, data_type, data["id"]+".png")
            # move image file
            os.rename(data["image"], new_image_path)
            # modify image path in data
            data["image"] = new_image_path
            # add to new jsons
            if data_type == "general":
                lst_general.append(data)
            elif data_type == "regional":
                lst_regional.append(data)
            elif data_type == "suggestion":
                lst_suggestion.append(data)


    with open(os.path.join(args.image_folder, "general.json"), "w") as f:
        json.dump(lst_general, f, indent=4)
    with open(os.path.join(args.image_folder, "regional.json"), "w") as f:
        json.dump(lst_regional, f, indent=4)
    with open(os.path.join(args.image_folder, "suggestion.json"), "w") as f:
        json.dump(lst_suggestion, f, indent=4)


    