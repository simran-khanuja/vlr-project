import os
import argparse
import yaml
import random
import logging
import cv2
import json
import requests
from io import BytesIO
import PIL
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

def download_image(path):
    # check if image is URL and download if yes
    if path.startswith("http"):
        response = requests.get(path, timeout=120)
        if response.status_code == 200 and response.headers['Content-Type'].startswith('image'):
            image = PIL.Image.open(BytesIO(response.content))
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        else:
            logging.info(f"Invalid response")
            return "error"
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def concatenate_images(source_img, target_img, save_path):
    total_width = source_img.width + target_img.width
    max_height = max(source_img.height, target_img.height)
    concatenated_img = Image.new('RGB', (total_width, max_height))
    concatenated_img.paste(source_img, (0, 0))
    concatenated_img.paste(target_img, (source_img.width, 0))
    concatenated_img.save(save_path)

def resize_image(image, threshold_size=1024):
    w, h = image.size
    if w > threshold_size or h > threshold_size:
        if w > h:
            new_w = threshold_size
            new_h = int(h * (threshold_size / w))
        else:
            new_h = threshold_size
            new_w = int(w * (threshold_size / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

def convert_to_canny(image, low_threshold=100, high_threshold=200):
    image_array = np.array(image)
    canny_image = cv2.Canny(image_array, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    return canny_image

def create_controlnet_pipeline(device):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False

def main():
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Read in config file to get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlnet.yaml", help="Path to config file.")
   
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    random.seed(config["seed"])
    
    # Initialize ControlNet
    pipe = create_controlnet_pipeline(device)
    pipe.safety_checker = disabled_safety_checker

    # Get source images
    source_countries_list = config["source_countries"]
    source_data_path = config["source_data_path"]

    all_source_paths = []
    all_source_countries = []
    for country in source_countries_list:
        country_paths_file = source_data_path + "/" + country + ".json"
        with open(country_paths_file) as f:
            data = json.load(f)
            # get values from json file which is a dictionary of dictionaries
            for category in data:
                all_source_paths.extend(data[category].values())
                all_source_countries.extend([country] * len(data[category].values()))

    if config["debug"]:
        logging.info("Debug mode enabled. Using 20 random images.")
        all_source_paths = random.sample(all_source_paths, 20)
        all_source_countries = random.sample(all_source_countries, 20)
    logging.info("Number of images: " + str(len(all_source_paths)))
    
    # Iterate over each image path and remove it if it doesn't exist
    source_paths = []
    source_countries = []

    for i in range(len(all_source_paths)):
        #if os.path.exists(all_source_paths[i]):
        source_paths.append(all_source_paths[i])
        source_countries.append(all_source_countries[i])
    logging.info("Number of images: " + str(len(source_paths)))

    num_inference_steps = int(config["num_inference_steps"])
    guidance_scale = float(config["text_guidance"])

    # output_dir = config["output_dir"] + "/" + str(num_inference_steps) + "_" + str(image_guidance_scale) + "_" + str(guidance_scale)
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    with open(output_dir + "/metadata.csv", "w") as f:
        f.write("src_image_path,src_country,tgt_image_path,prompt\n")
        for i, image_path in enumerate(source_paths):
            try:
                image = download_image(image_path)
                image = resize_image(image)
                prompt = config["prompt"]
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                #generator = [torch.Generator(device="cuda").manual_seed(2)]
                src_country = source_countries[i]
                canny_image = convert_to_canny(image)
                generated_image = pipe(
                    prompt,
                    canny_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                generated_image_path = output_dir + "/" + src_country + "_" + image_path.split("/")[-1]
                generated_image.save(generated_image_path)
                f.write(image_path + "," + source_countries[i] + "," + generated_image_path + "," + prompt + "\n")
            except torch.cuda.OutOfMemoryError as e:
                logging.info(f"Skipping image {image_path} due to CUDA OOM error: {e}")
                continue

if __name__ == "__main__":
    main()