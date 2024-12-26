import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import gc

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def question_sort(questions):
    q_general, q_regional, q_suggestion = [], [], []
    for q in questions:
        if q["id"].split("_")[1] == "general":
            q_general.append(q)
        elif q["id"].split("_")[1] == "regional":
            q_regional.append(q)
        elif q["id"].split("_")[1] == "suggestion":
            q_suggestion.append(q)
    
    sorted_questions = {"general":q_general, "regional":q_regional, "suggestion":q_suggestion}
    return sorted_questions


def eval_model(args):

    # read and sort questions
    with open(os.path.expanduser(args.question_file), 'r') as f:
        questions = json.load(f)
    questions = question_sort(questions)

    # prepare output
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    output_result = {}
    if args.modify_mode:
        with open(args.old_answers, "r") as f:
            output_result = json.load(f)

    q_types = [("general", args.general_ckpt), ("regional", args.regional_ckpt), ("suggestion", args.suggestion_ckpt)]
    q_types = [q_types[ids] for ids in args.qtype_choices]
    # Model: general/regional/suggestion
    for qtype in q_types:
        disable_torch_init()
        model_path = os.path.expanduser(qtype[1])
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(qtype[1], args.model_base, model_name)

        print("Evaluating question type, ", qtype[0])
        for line in tqdm(questions[qtype[0]]):
            idx = line["id"]
            image_file = line["image"]
            qs = line["conversations"][0]['value']
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            output_result[idx] = outputs
            print(prompt)
            print(outputs)
            print("=================")

        # free cuda memory
        model.cpu()
        del tokenizer, model, image_processor
        gc.collect()
        torch.cuda.empty_cache()

    with open(answers_file, "w") as file:
        json.dump(output_result, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_ckpt", type=str, default="../llava-v1.5-7b-task-lora-general/")
    parser.add_argument("--regional_ckpt", type=str, default="../llava-v1.5-7b-task-lora-regional/")
    parser.add_argument("--suggestion_ckpt", type=str, default="../llava-v1.5-7b-task-lora-suggestion/")
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="../dataset/test_detected.json")
    parser.add_argument("--answers-file", type=str, default="../submission.json")
    parser.add_argument("--modify_mode", type=bool, default=False)
    parser.add_argument("--old_answers", type=str, default="../old_submission.json")
    parser.add_argument("--qtype_choices", type=list, default=[0,1,2])
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
