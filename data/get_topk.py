import json
import random
import os
import PIL
from PIL import Image
from io import BytesIO
import requests
import argparse
import yaml
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel


def download_image(path):
    if path.startswith("http"):
        response = requests.get(path, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Read in config file to get args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/topk.yaml", help="Path to config file.")
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if not os.path.exists(config["output_dir"]):
    os.makedirs(config["output_dir"], exist_ok=True)

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained(config["model"])
processor = CLIPProcessor.from_pretrained(config["model"])

# Read multiple jsonl files based on dataset_stats.txt
input_dir = config["input_dir"]
countries = config["countries"]
files = [country + "_dataset.jsonl" for country in countries]

image_data = []

# Get random subset of images from each country
for file, country in zip(files, countries):
    with open(input_dir + "/" + file, 'r') as f:
        print(file)
        data = [json.loads(line) for line in f.readlines()]

    if len(data) >= config["size_per_country"]:
        image_data.extend(random.sample(data, config["size_per_country"]))
    else:
        image_data.extend(data)

    image_paths = []
    filtered_data = []
    for i, data in enumerate(image_data):
        # Check if image gets downloaded
        try:
            # Get the image from the URL
            response = requests.get(data['url'], timeout=15)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # also try processing the image
            _ = processor(text=None, images=[download_image(data['url'])], return_tensors="pt", padding=True).cuda()
        except:
            continue
        image_paths.append(data['url'])
        filtered_data.append(data)

    batch_size = 16
    clip_similarities = []
    # Run CLIP model
    for i in range(0, len(image_paths), batch_size):
        images_batch = [download_image(image_path) for image_path in image_paths[i:i+batch_size]]
        prompts_batch = [config["prompt"]] * len(images_batch)
        prompt_features = model.get_text_features(**prompts_batch).cuda()
            
        image_inputs = processor(text=None, images=images_batch, return_tensors="pt", padding=True).cuda()
        image_features = model.get_image_features(**image_inputs)
            
        # Calculate similarity
        clip_similarities.extend(F.cosine_similarity(prompt_features, image_features).cpu().tolist())
    
    # sort similarities and get top k indices
    topk_indices = sorted(range(len(clip_similarities)), key=lambda i: clip_similarities[i], reverse=True)[:config["topk"]]

    # Write topk data to file
    with open(os.path.join(config["output_dir"], country + "_topk.jsonl"), "w") as f:
        for i in topk_indices:
            f.write(json.dumps(filtered_data[i]) + "\n")

    # Write similarities to file
    with open(os.path.join(config["output_dir"], country + "_clip_sim.txt"), "w") as f:
        for path, similarity in zip(image_paths, clip_similarities):
            f.write(f"{path}\t{similarity}\n")

    print(f"Finished processing {country}.")



