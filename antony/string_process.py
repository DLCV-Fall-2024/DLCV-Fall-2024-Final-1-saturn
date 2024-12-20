from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import json
import os
from tqdm import tqdm
import numpy as np
from torchvision.ops import box_convert
import torch

'''
print(xyxy)
                    bbox_color = (255, 0, 0)  # Blue color in BGR
                    thickness = 2

                    # Draw rectangle
                    scene_depth_map = scene_depth_info.copy()/600
                    scene_depth_map = scene_depth_map*255
                    depth_image_with_bbox = cv2.rectangle(scene_depth_map.astype(np.uint8), (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), bbox_color, thickness)
                    print(np.max(scene_depth_map))
                    print(np.min(scene_depth_map))

                    # Display the image
                    output_path = 'depth_image_with_bbox.png'
                    cv2.imwrite(output_path, depth_image_with_bbox)
                    asd
'''

if __name__ == "__main__":
    splits = ['train','val','test']
    output_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/detection_info"
    info_root = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/detection_info"
    meta_information_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset"
    for split in splits:
        output_path = os.path.join(output_dir,f"{split}_combined.json")
        output_data = {}
        meta_information_path = os.path.join(meta_information_dir,f"{split}.json")
        detect_filename = os.path.join(info_root,f"{split}_detection.json")
        depth_dir = os.path.join(info_root,'depth_info')
        depth_dir = os.path.join(depth_dir,split)
        with open(meta_information_path, 'r') as file:
            meta_datas = json.load(file)
        with open(detect_filename, 'r') as file:
            detect_info = json.load(file)
        for meta_data in tqdm(meta_datas):
            combined_info = {}
            scene_name = meta_data['id']
            scene_detect_info = detect_info[scene_name] # xyxy
            scene_depth_info_origin = np.load(os.path.join(depth_dir,f"{scene_name}.npy"))
            scene_depth_map_vis = scene_depth_info_origin.copy()/np.max(scene_depth_info_origin)
            scene_depth_map_vis = scene_depth_map_vis*255
            scene_depth_map_vis = scene_depth_map_vis.astype(np.uint8)
            # Draw the 
            for category in scene_detect_info.keys():
                combined_info[category] = []
                for bbox in scene_detect_info[category]:
                    scene_depth_info = scene_depth_info_origin.copy()
                    instance = {}
                    h, w = scene_depth_info.shape
                    bbox = torch.tensor(bbox) * torch.Tensor([w, h, w, h])
                    # xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                    xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.int32)
                    instance['bounding_box'] = xyxy.tolist()
                    x_min,y_min,x_max,y_max = xyxy
                    # Use one std's depth's mean as this object's depth
                    bbox_depth_values = scene_depth_info[y_min:y_max, x_min:x_max].flatten()

                    # Step 2: Compute the mean and standard deviation of depth values
                    mean_depth = np.mean(bbox_depth_values)
                    std_depth = np.std(bbox_depth_values)

                    # Step 3: Filter values within one standard deviation of the mean
                    filtered_depth_values = bbox_depth_values[
                        (bbox_depth_values >= mean_depth - std_depth) & 
                        (bbox_depth_values <= mean_depth + std_depth)
                    ]

                    # Step 4: Calculate the mean depth from the filtered values
                    object_depth = np.mean(filtered_depth_values)
                    min_depth = np.min(scene_depth_info)
                    max_depth = np.max(scene_depth_info)
                    ratio = (object_depth - min_depth)/ (max_depth - min_depth)
                    if(ratio>=0.75):
                        instance['range'] = "immediate"
                    elif(ratio>=0.50):
                        instance['range'] = "short range"
                    elif(ratio>=0.25):
                        instance['range'] = "mid range"
                    else:
                        instance['range'] = "long range"
                    combined_info[category].append(instance)
                    # bbox_color = (255, 0, 0)  # Blue color in BGR
                    # thickness = 2

                    # Draw rectangle
                    # scene_depth_map_vis = cv2.rectangle(scene_depth_map_vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), bbox_color, thickness)
                    # Display the image
                # output_path_img = f'depth_image_with_bbox_{split}.png'
                # cv2.imwrite(output_path_img, scene_depth_map_vis)    
            output_data[scene_name] = combined_info
        with open(output_path, "w") as file:
            json.dump(output_data, file, indent=4)

