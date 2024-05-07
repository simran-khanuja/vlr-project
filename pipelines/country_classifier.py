import json
import random
import os
import PIL
from PIL import Image
from io import BytesIO
import requests
import csv
import argparse
import yaml
from collections import defaultdict
import urllib3

from transformers import CLIPProcessor, CLIPModel


def download_image(path):
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Read in config file to get args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/clip-classifier.yaml", help="Path to config file.")
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if not os.path.exists(config["output_image_folder"]):
    os.makedirs(config["output_image_folder"], exist_ok=True)

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Read multiple jsonl files based on dataset_stats.txt
input_dir = config["input_dir"]
with open(config["dataset_stats_file"], 'r') as stats_file:
    entries = [line.split() for line in stats_file.readlines()]
    # get top 100 countries
    top_countries = [' '.join(entry[:-1]) for entry in entries][:config["number_of_countries"]]

top_files = [country + "_dataset.jsonl" for country in top_countries]

image_data = []
countries=[]

# Get random subset of images from each country
for file, country in zip(top_files, top_countries):
    with open(input_dir + "/" + file, 'r') as f:
        print(file)
        data = [json.loads(line) for line in f.readlines()]

        if len(data) >= config["size_per_country"]:
            image_data.extend(random.sample(data, config["size_per_country"]))
            countries.extend([country] * config["size_per_country"])

# Download images and create prompts
prompts_set = set()
labels = []
image_paths = []
for (i, data), country in zip(enumerate(image_data), countries):
    if i%100 == 0:
        print(i)
    try:
        # Get the image from the URL
        response = requests.get(data['URL'], timeout=15)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Open the image using PIL
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        save_image_path = config["output_image_folder"] + f"/{country}_{i}.jpg"
        
        # Save or process the image (here, we're saving it)
        image.save(save_image_path)
        prompts_set.add(f"this is a photo from {country}")
        labels.append(f"this is a photo from {country}")
        image_paths.append(save_image_path)
        
    except (requests.RequestException, Image.UnidentifiedImageError, \
        urllib3.exceptions.ProtocolError, urllib3.exceptions.ReadTimeoutError):
        continue

prompts = list(prompts_set)

batch_size = 16
results = []
# (e) Run CLIP model
for i in range(0, len(image_paths), batch_size):
    # images = [Image.open(requests.get(data['URL'], stream=True, timeout=15).raw) for data in sampled_image_data[i:i+batch_size]]
    images = [download_image(image_path) for image_path in image_paths[i:i+batch_size]]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    for prob in probs:
        result = [
            {"score": score, "label": prompt}
            for score, prompt in sorted(zip(prob, prompts), key=lambda x: -x[0])
        ]
        results.append(result)


# Write to CSV and highest scoring pred to metadata.csv
with open(config["output_image_folder"] + '/output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image_Path', 'Probabilities', 'Label'])
    for path, prob, label in zip(image_paths, results, labels):
        writer.writerow([path, prob, label])

with open(config["output_image_folder"] + '/metadata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image', 'caption'])
    for path, result, label in zip(image_paths, results, labels):
        text = "Prediction: " + result[0]["label"].replace("this is a photo from ", "") + "; Label: " + label.replace("this is a photo from ", "")
        writer.writerow([path, text])