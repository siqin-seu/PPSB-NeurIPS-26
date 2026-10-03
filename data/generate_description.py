import openai
from openai import OpenAI
import os
import json
from tqdm import tqdm
from pathlib import Path

client = OpenAI(
    api_key="Your-keys",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

category_name_path = {
    'CUB' : './data_/CUB/CUB_200_2011/classes.txt',
    'DTD' : './data_/DTD/class_names.txt',
    'FGVCAircraft' : './data_/FGVCAircraft/labels.txt',
    'Flowers102' : './data_/Flowers102/class_names.txt',
    'OxfordPets' : './data_/OxfordPets/class_names.txt',
    'StanfordCars' : './data_/StanfordCars/class_names.txt',
}

dataset_custom_prompts = {
    'DTD' : 'a photo of {} {} texture',
    'FGVCAircraft' : 'a photo of {} {}, a type of aircraft',
    'Flowers102' : 'a photo of {} {}, a type of flower',
    'CUB' : 'a photo of {} {}, a type of bird',
    'OxfordPets' : 'a photo of {} {}, a type of pet',
    'StanfordCars' : 'a photo of {} {}, a type of car',
}

def get_labels_list(dataset, label_path):

    if dataset == "CUB":

        classes = []
        with open(label_path, "r") as f:
            for l in f:
                classes.append(l.strip().split(".")[-1].replace("_", " ").lower())

    elif dataset == "DTD":

        classes = []
        with open(label_path, "r") as f:
            for l in f:
                classes.append(l.strip())

    elif dataset == "FGVCAircraft":

        classes = []
        with open(label_path, "r") as f:
            for l in f:
                classes.append(l.strip())

    elif dataset == "Flowers102":

        classes = []
        with open(label_path, "r") as f:
            for l in f:
                classes.append(l.strip())

    elif dataset == "OxfordPets":

        classes = []
        with open(label_path, "r") as f:
            for l in f:
                classes.append(l.strip())
    
    elif dataset == "StanfordCars":

        classes = []
        with open(label_path, "r") as f:
            for l in f:
                classes.append(l.strip())

    return classes

vowel_list = ['A', 'E', 'I', 'O', 'U']

Path(f"generic").mkdir(parents=True, exist_ok=True)

for k, v in category_name_path.items():
    all_json_dict = {}
    all_responses = {}
    print('Generating descriptions for ' + k + ' dataset.')
    json_name_all = f"generic/{k}.json"

    if Path(json_name_all).is_file():
        raise ValueError("File already exists")
    
    labels_name = get_labels_list(k, v)

    for category in labels_name:
        print(category)
        if category.upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        if '_' in category:
            cat = category.replace('_', ' ')
        else:
            cat = category

        prompts = []
        # prompts.append("Describe what " + dataset_custom_prompts[k].format(article, cat) + " looks like")
        prompts.append("How to identify " + dataset_custom_prompts[k].format(article, cat) + "?")
        # prompts.append("What does " + article + " " + cat + " look like?")
        # prompts.append("Describe an image from the internet of " + article + " " + cat)
        # prompts.append("A caption of an image of " + article + " " + cat + ":")

        system_prompt = "You are a helpful assistant for describing the visual characteristics of the given category." \
        "Describe the unique features of each category instead of their common features. Please describe in a concise sentence." \
        "Use the following template : a photo of [class] [unique features]."
        
        all_result = []
        for curr_prompt in prompts:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    { "role": "user", "content": curr_prompt },
                    { "role": "system", "content": system_prompt }
                ],
                temperature=.7,
                max_tokens=55,
                n=1,
                stop="."
            )

            for r in range(len(response.choices)):
                result = response.choices[r].message.content
                all_result.append(result.replace("\n\n", "") + ".")

        all_responses[cat] = all_result

    with open(json_name_all, 'w') as f:
        json.dump(all_responses, f, indent=4)