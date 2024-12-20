import json
import os
from tqdm import tqdm
import numpy as np
import copy

def construct_pool(pool_name):
    pool_scene_name_general,pool_scene_name_regional,pool_scene_name_general_suggestion = [],[],[]
    pool_name_general,pool_name_regional,pool_name_suggestion = [],[],[]
    for i in pool_name:
        name = os.path.basename(i).split(".npy")[0]
        if('general' in i):
            pool_scene_name_general.append(name)
            pool_name_general.append(i)
        elif('regional' in i):
            pool_scene_name_regional.append(name)
            pool_name_regional.append(i)
        else:
            pool_scene_name_general_suggestion.append(name)
            pool_name_suggestion.append(i)
    pool_general = [np.load(i) for i in tqdm(pool_name_general)]
    pool_general = np.array(pool_general)
    pool_general = pool_general / np.linalg.norm(pool_general)
    pool_regional = [np.load(i) for i in tqdm(pool_name_regional)]
    pool_regional = np.array(pool_regional)
    pool_regional = pool_regional / np.linalg.norm(pool_regional)
    pool_suggestion = [np.load(i) for i in tqdm(pool_name_suggestion)]
    pool_suggestion = np.array(pool_suggestion)
    pool_suggestion = pool_suggestion / np.linalg.norm(pool_suggestion)
    return pool_general,pool_regional,pool_suggestion,pool_scene_name_general,pool_scene_name_regional,pool_scene_name_general_suggestion

def find_relevant_k_scenario(pool,pool_scene_name,k,query_scene_name,query_pool,query_pool_scene_name):
    index = query_pool_scene_name.index(query_scene_name)
    query = query_pool[index].copy()
    print(query_scene_name)
    cosine_similarity = np.dot(pool, query)

    # Get the indices of the top-k most related features
    top_k_indices = np.argsort(cosine_similarity)[-k:][::-1]  # Sort and reverse for descending order

    # Retrieve top-k features
    # top_k_features = pool[top_k_indices]
    pool_scene_name_np = np.array(pool_scene_name)
    print(pool_scene_name_np[top_k_indices])
    print("======================")
    return pool_scene_name_np[top_k_indices]
    # print(f"Query scene name {query_scene_name}")
    # print(f"Indices of Top-{k} Features:", top_k_indices)
    # pool_scene_name = np.array(pool_scene_name)
    # print(f"Top-{k} Features:\n", pool_scene_name[top_k_indices])
    # asd
    
def find_element_by_id(data_list, target_ids):
    result = []
    for target_id in target_ids:
        for element in data_list:
            if element["id"] == target_id:
                result.append(element["conversations"][1]['value'])
    return result

def construct_v1_text(ori_task_description,combined_data,scene_name):
    '''
    GT answer:
    In the observed traffic scene, there are several notable road users and objects that are crucial for the autonomous navigation of the ego car.
    Firstly, there is a large red truck on the left lane, slightly ahead of the ego car and traveling in the same direction. 
    This truck, which is carrying multiple cars, poses a visibility concern and necessitates careful monitoring due to its size and the potential impact on safe following distances.
    An adjustment in speed might be required to maintain safety.\n\n
    Additionally, there are several cars in front of the ego car, occupying both the center and right lanes, which indicates a moderate level of traffic flow. 
    The positioning of these vehicles is particularly important as it influences the ego car's ability to change lanes safely and maintain an appropriate distance from the vehicles ahead.\n\n
    Ahead of the traffic, there is a traffic light displaying a green signal for going straight and a green arrow for making left turns. 
    This indicates that the ego car can proceed without the necessity to stop, allowing for continued movement through the intersection or a turn, based on the intended route.\n\n
    Furthermore, yellow and black traffic cones are positioned on the right side of the road, signaling ongoing roadside work or the presence of a hazard. This setup warrants cautious driving from the ego car, which might include reducing speed or maneuvering carefully to avoid the cones and any potential hazards associated with the indicated roadwork or obstacle.\n\n
    In this scenario, other possible elements such as vulnerable road users, additional traffic signs, barriers, and miscellaneous objects are not present or deemed relevant. The absence of these elements simplifies the autonomous driving decisions to be made but nonetheless requires attentiveness to the described road users and traffic controls.
    '''
    prompt = ""
    # step 1 (Task Description)
    prompt += "\tAnalyze the provided image captured from the perspective of the ego car in a traffic scenario. Focus exclusively on objects that directly influence the ego car's driving behavior. These objects are categorized into seven groups: vehicles (e.g., cars, trucks, buses), vulnerable road users (e.g., pedestrians, cyclists, motorcyclists), traffic signs (e.g., no parking, warning, directional, regulatory), traffic lights (e.g., red, yellow, green), traffic cones and barriers (temporary or permanent obstructions), and miscellaneous hazards (e.g., debris, dustbins, animals). Objects outside these categories should not be described.For each detected object, provide a detailed analysis covering its appearance, position, and direction. The appearance should describe the object's visual characteristics, while the position specifies its relative location to the ego car (e.g., in the left lane or at the edge of the road). The direction should indicate the object’s movement, if applicable (e.g., stationary, approaching, or moving away). Additionally, explain how each object impacts the ego car's driving behavior. This explanation should include safety considerations, such as potential hazards or collision risks, adjustments to lane positioning to avoid the object, and driving decisions like changes in speed, direction, or the need to stop. Prioritize detail and relevance in your analysis, focusing on how the identified objects affect the ego car's ability to navigate safely and effectively.\n"
    # Step 2 (study (Not in this version) )
    pass
    # Step 3 (Target image analysis)
    prompt += "\tTARGET IMAGE: <image>\n The detection results for the target image are as follows, categorized by object type:\n"
    prompt += f"{str(combined_data)}\n"
    prompt += "Object Descriptions and Explanations: For each detected object, provide a detailed description based on the following structure:\
                1. Appearance: Describe the object’s color, type (e.g.,\
                white parked vehicles, pedestrian with a yellow jacket, orange traffic cones), and any distinguishing features (e.g.,\
                debris, construction materials).\
                2. Position: Specify the object’s location relative to the ego\
                car (e.g., parked to the left, walking alongside the road,\
                marking the right edge of the lane).\
                3. Direction: Indicate the object’s direction of movement\
                (e.g., stationary, walking parallel to the road).\
                4. Impact on Ego Car: Explain how this object affects the\
                ego car’s driving behavior.\
                Consider factors such as pedestrians crossing the road,\
                parked vehicles reducing lane width, and construction zones\
                requiring lane adjustments.\
                - Traffic Control Devices and Signs: Describe traffic\
                signs and their effect on driving behavior (e.g., pedestrian\
                crossings requiring caution). If signs are blurred or not\
                clearly visible, note their presence but clarify that they\
                cannot be used to guide driving decisions.\
                - Other Objects: Mention environmental objects like debris or roadside objects. If they are not an immediate hazard but could potentially become one, describe how the\
                ego vehicle should monitor them while passing.\
                - Scenario Summary: After describing individual objects,\
                generate a comprehensive summary of the scene. Integrate all relevant objects by category and explain their\
                collective influence on the ego car’s behavior, focusing\
                on how these elements contribute to the overall traffic environment and decision-making process.\n"
    # Step 4  Suppression of Hallucinations
    prompt += "\tFocus solely on the objects detected in the target image.\
                Avoid adding any object descriptions or explanations that\
                are not explicitly mentioned in the detection results.\
                - Hallucinations occur when objects not present in the detection results are erroneously included in the response.\
                Before finalizing your description of any object, confirm\
                that it is based on the detection results.\
                - If a detected object category is more abstract (e.g., ‘car’\
                for an SUV), use contextual visual cues from the image to\
                provide more detailed descriptions where appropriate, but\
                do not introduce new objects that are not clearly visible.\
                - For objects like flashing lights or reflective surfaces\
                (e.g., vehicle rear lights), ensure that the interpretation is\
                grounded in detected data, and provide a plausible explanation based on context (e.g., rear lights flashing during braking)."
   
    return prompt

def construct_v2_text(ori_task_description,combined_data,scene_name,top_k_response):
    '''
    GT answer:
    In the observed traffic scene, there are several notable road users and objects that are crucial for the autonomous navigation of the ego car.
    Firstly, there is a large red truck on the left lane, slightly ahead of the ego car and traveling in the same direction. 
    This truck, which is carrying multiple cars, poses a visibility concern and necessitates careful monitoring due to its size and the potential impact on safe following distances.
    An adjustment in speed might be required to maintain safety.\n\n
    Additionally, there are several cars in front of the ego car, occupying both the center and right lanes, which indicates a moderate level of traffic flow. 
    The positioning of these vehicles is particularly important as it influences the ego car's ability to change lanes safely and maintain an appropriate distance from the vehicles ahead.\n\n
    Ahead of the traffic, there is a traffic light displaying a green signal for going straight and a green arrow for making left turns. 
    This indicates that the ego car can proceed without the necessity to stop, allowing for continued movement through the intersection or a turn, based on the intended route.\n\n
    Furthermore, yellow and black traffic cones are positioned on the right side of the road, signaling ongoing roadside work or the presence of a hazard. This setup warrants cautious driving from the ego car, which might include reducing speed or maneuvering carefully to avoid the cones and any potential hazards associated with the indicated roadwork or obstacle.\n\n
    In this scenario, other possible elements such as vulnerable road users, additional traffic signs, barriers, and miscellaneous objects are not present or deemed relevant. The absence of these elements simplifies the autonomous driving decisions to be made but nonetheless requires attentiveness to the described road users and traffic controls.
    '''

    prompt = ""
    # step 1 (Task Description)
    prompt += "\tAnalyze the provided image captured from the perspective of the ego car in a traffic scenario. Focus exclusively on objects that directly influence the ego car's driving behavior. These objects are categorized into seven groups: vehicles (e.g., cars, trucks, buses), vulnerable road users (e.g., pedestrians, cyclists, motorcyclists), traffic signs (e.g., no parking, warning, directional, regulatory), traffic lights (e.g., red, yellow, green), traffic cones and barriers (temporary or permanent obstructions), and miscellaneous hazards (e.g., debris, dustbins, animals). Objects outside these categories should not be described.For each detected object, provide a detailed analysis covering its appearance, position, and direction. The appearance should describe the object's visual characteristics, while the position specifies its relative location to the ego car (e.g., in the left lane or at the edge of the road). The direction should indicate the object’s movement, if applicable (e.g., stationary, approaching, or moving away). Additionally, explain how each object impacts the ego car's driving behavior. This explanation should include safety considerations, such as potential hazards or collision risks, adjustments to lane positioning to avoid the object, and driving decisions like changes in speed, direction, or the need to stop. Prioritize detail and relevance in your analysis, focusing on how the identified objects affect the ego car's ability to navigate safely and effectively.\n"
    # Step 2 (study (Not in this version) )
    pass
    study_prompt = "\tBefore analyzing the target image, carefully study the following two example detection results and their corresponding outputs. These examples illustrate the expected level\
of detail and structure. Your goal is to replicate this structure by providing accurate object descriptions with a clear\
focus on appearance, position, direction, and the impact\
each object has on the ego car’s driving behavior, specifically noting the object’s category.\
Example General Perception Output"
    for example in top_k_response:
        study_prompt += "1: {" + f"{example}" + "}"
    study_prompt += "IMPORTANT: Your output must follow the structure\
demonstrated in the examples. Ensure that every detected\
object is described precisely with details on appearance, position, and direction, followed by a well-reasoned explanation of its impact on the ego car’s behavior. Avoid including\
objects not present in the detection results.\n"
    prompt += study_prompt
    # Step 3 (Target image analysis)
    prompt += "\tTARGET IMAGE: <image>\n The detection results for the target image are as follows, categorized by object type:\n"
    prompt += f"{str(combined_data)}\n"
    prompt += "Object Descriptions and Explanations: For each detected object, provide a detailed description based on the following structure:\
1. Appearance: Describe the object’s color, type (e.g.,\
white parked vehicles, pedestrian with a yellow jacket, orange traffic cones), and any distinguishing features (e.g.,\
debris, construction materials).\
2. Position: Specify the object’s location relative to the ego\
car (e.g., parked to the left, walking alongside the road,\
marking the right edge of the lane).\
3. Direction: Indicate the object’s direction of movement\
(e.g., stationary, walking parallel to the road).\
4. Impact on Ego Car: Explain how this object affects the\
ego car’s driving behavior.\
Consider factors such as pedestrians crossing the road,\
parked vehicles reducing lane width, and construction zones\
requiring lane adjustments.\
- Traffic Control Devices and Signs: Describe traffic\
signs and their effect on driving behavior (e.g., pedestrian\
crossings requiring caution). If signs are blurred or not\
clearly visible, note their presence but clarify that they\
cannot be used to guide driving decisions.\
- Other Objects: Mention environmental objects like debris or roadside objects. If they are not an immediate hazard but could potentially become one, describe how the\
ego vehicle should monitor them while passing.\
- Scenario Summary: After describing individual objects,\
generate a comprehensive summary of the scene. Integrate all relevant objects by category and explain their\
collective influence on the ego car’s behavior, focusing\
on how these elements contribute to the overall traffic environment and decision-making process.\n"
    # Step 4  Suppression of Hallucinations
    prompt += "\tFocus solely on the objects detected in the target image.\
Avoid adding any object descriptions or explanations that\
are not explicitly mentioned in the detection results.\
- Hallucinations occur when objects not present in the detection results are erroneously included in the response.\
Before finalizing your description of any object, confirm\
that it is based on the detection results.\
- If a detected object category is more abstract (e.g., ‘car’\
for an SUV), use contextual visual cues from the image to\
provide more detailed descriptions where appropriate, but\
do not introduce new objects that are not clearly visible.\
- For objects like flashing lights or reflective surfaces\
(e.g., vehicle rear lights), ensure that the interpretation is\
grounded in detected data, and provide a plausible explanation based on context (e.g., rear lights flashing during braking)."

    return prompt


if __name__ == "__main__":
    # v1 (version 1) -> Encode detection and depth information (General)
    # v2 (version 2) -> Encode detection and depth information + Few shot learning (General)
    # v3 (version 3) -> Encode detection and depth information + Few shot learning (General + Regional)
    # v4 (version 3) -> Encode detection and depth information + Few shot learning (General + Regional + Suggestion)
    version = "v2"
    splits = ['train','val','test']
    output_dir = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset"
    combined_root = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/detection_info"
    pool_root = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/pool"
    val_refer_data_path = "/home/antony/DLCV-Fall-2024-Final-1-saturn/dataset/val.json"
    with open(val_refer_data_path, 'r') as file:
        val_refer_datas = json.load(file)
    k = 2 # Follow their work
    if(version != "v1"):
        pool_train_dir = os.path.join(pool_root,'train')
        pool_val_dir = os.path.join(pool_root,'val')
        pool_test_dir = os.path.join(pool_root,'test')
        pool_name = [ os.path.join(pool_train_dir,i) for i in os.listdir(pool_train_dir) ]
        pool_name.extend([ os.path.join(pool_val_dir,i) for i in os.listdir(pool_val_dir) ])
        query_pool_name = [ os.path.join(pool_test_dir,i) for i in os.listdir(pool_test_dir) ]
        query_pool_name.extend(copy.deepcopy(pool_name))
        pool_general,pool_regional,pool_suggestion,pool_scene_name_general,pool_scene_name_regional,pool_scene_name_general_suggestion = construct_pool(pool_name)
        query_pool_general,query_pool_regional,query_pool_suggestion,query_pool_scene_name_general,query_pool_scene_name_regional,query_pool_scene_name_general_suggestion = construct_pool(query_pool_name)
        


    for split in tqdm(splits):
        if(version == "v1"):
            output_path = os.path.join(output_dir,f"{split}_v1.json")
            refer_path = os.path.join(output_dir,f"{split}.json")
            combined_path = os.path.join(combined_root,f"{split}_combined.json")
            output_data = []
            with open(refer_path, 'r') as file:
                refer_datas = json.load(file)
            with open(combined_path, 'r') as file:
                combined_datas = json.load(file)
            for refer_data in refer_datas:
                scene_data = {}
                combined_data = combined_datas[refer_data['id']]
                scene_data["id"] = refer_data['id']
                scene_data["image"] = refer_data['image']
                scene_data["conversations"] = refer_data['conversations']
                if("general" in refer_data['id']):
                    scene_data["conversations"][0]['value'] = construct_v1_text(scene_data["conversations"][0]['value'],combined_data,scene_data["id"])
                output_data.append(scene_data)
        if(version == "v2"):
            output_path = os.path.join(output_dir,f"{split}_{version}.json")
            refer_path = os.path.join(output_dir,f"{split}.json")
            combined_path = os.path.join(combined_root,f"{split}_combined.json")
            output_data = []
            with open(refer_path, 'r') as file:
                refer_datas = json.load(file)
            with open(combined_path, 'r') as file:
                combined_datas = json.load(file)
            refer_datas_retireve = copy.deepcopy(refer_datas)
            refer_datas_retireve.extend(val_refer_datas)
            refer_datas_retireve = np.array(refer_datas_retireve)
            
            for refer_data in tqdm(refer_datas):
                scene_data = {}
                combined_data = combined_datas[refer_data['id']]
                scene_data["id"] = refer_data['id']
                scene_data["image"] = refer_data['image']
                scene_data["conversations"] = refer_data['conversations']
                
                if("general" in refer_data['id']):
                    top_k_scene_name = find_relevant_k_scenario(pool_general,pool_scene_name_general,k,refer_data['id'],query_pool_general,query_pool_scene_name_general)
                    top_k_response = find_element_by_id(refer_datas_retireve,top_k_scene_name)
                    scene_data["conversations"][0]['value'] = construct_v2_text(scene_data["conversations"][0]['value'],combined_data,scene_data["id"],top_k_response)
                output_data.append(scene_data)
        with open(output_path, "w") as json_file:
            json.dump(output_data, json_file, indent=4)
            
