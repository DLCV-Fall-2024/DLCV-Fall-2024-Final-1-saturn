from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import json
import os
from tqdm import tqdm

model = load_model("/home/antony/DLCV-Fall-2024-Final-1-saturn/antony/NexusAD/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/home/antony/DLCV-Fall-2024-Final-1-saturn/antony/NexusAD/GroundingDINO/weights/groundingdino_swinb_ogc.pth")
TEXT_PROMPT = "vehicles . vulnerable_road_users . traffic signs . traffic lights . traffic cones . barriers . other objects"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
splits = ['train','val','test']
for split in splits:
    output_path = f"/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/detection_info/{split}_detection.json" 
    img_dir = f"/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/{split}"
    img_filenames = [ os.path.join(img_dir,i) for i in os.listdir(img_dir) ]
    result_json = {}
    for img_file in tqdm(img_filenames):
        instance = {}
        image_source, image = load_image(img_file)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        for box,phrase in zip(boxes,phrases):
            if(phrase not in instance):
                instance[phrase] = []
            instance[phrase].append(box.tolist())
        result_json[os.path.basename(img_file).split(".")[0]] = instance
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite("annotated_image.jpg", annotated_frame)

    with open(output_path, "w") as file:
        json.dump(result_json, file, indent=4)