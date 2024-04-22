import os
import yaml
import argparse
import pandas as pd
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    CLIPTokenizer,
    ViTImageProcessor,
    ViTModel,
)
import PIL
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader

import json
import requests
from io import BytesIO


def download_image(path):
    # check if image is URL and download if yes
    if path.startswith("http"):
        response = requests.get(path, timeout=120)
        if response.status_code == 200 and response.headers["Content-Type"].startswith("image"):
            image = PIL.Image.open(BytesIO(response.content))
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        else:
            # logging.info(f"Invalid response")
            return "error"
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read in config file to get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../configs/clip_score_country.yaml",
        help="Path to config file.",
    )
    parser.add_argument("--dino_weight", default=0.5, type=float, help="Weight for DINO similarity.")
    parser.add_argument("--clip_weight", default=0.5, type=float, help="Weight for CLIP similarity.")
    parser.add_argument("--output_dir", default="../output", help="Path to output directory.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for processing images.")
    parser.add_argument("--n_images", default="5", type=int, help="Number of images to use.")

    args = parser.parse_args()
    print(os.getcwd())
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # mkdir config["output_dir"] if it doesn't exist
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)

    data = pd.read_csv(config["metadata"])

    # Calculate clip similarity of source and target images with prompts
    # Load CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Load Dino model
    dino_processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb8")
    dino_model = ViTModel.from_pretrained("facebook/dino-vitb8").eval().to(device)

    base_path_india = "https://storage.googleapis.com/image-transcreation/vlr-project/food/larger-set/india"
    base_path_japan = "https://storage.googleapis.com/image-transcreation/vlr-project/food/larger-set/japan"
    all_source_image_paths = [f"{base_path_india}/{i}.jpg" for i in range(args.n_images)]
    all_target_image_paths = [f"{base_path_japan}/{i}.jpg" for i in range(args.n_images)]

    source_image_paths = []
    target_image_paths = []

    # check if paths are valid and remove paths that don't exist from both
    for i in range(len(all_source_image_paths)):
        # check if the image can be downloaded and processes
        print(all_source_image_paths[i])  # just to keep track of progress
        try:
            src_img = download_image(all_source_image_paths[i])
            tgt_img = download_image(all_target_image_paths[i])
            # also try processing the image for clip
            _ = clip_processor(
                text=None,
                images=[src_img],
                return_tensors="pt",
                padding=True,
            ).to(device)
            _ = clip_processor(
                text=None,
                images=[tgt_img],
                return_tensors="pt",
                padding=True,
            ).to(device)
            # also try processing the image for dino
            _ = dino_processor(
                src_img,
                return_tensors="pt",
                padding=True,
            ).to(device)
            _ = dino_processor(
                tgt_img,
                return_tensors="pt",
                padding=True,
            ).to(device)
            source_image_paths.append(all_source_image_paths[i])
            target_image_paths.append(all_target_image_paths[i])
        except:
            print(f"Source image path or target image path {all_target_image_paths[i]} cannot be downloaded. Removing from list.")
            continue

    # Read in prompt
    prompts = [config["prompt"]] * len(target_image_paths)

    # Calculate clip and dino features for all source and target images
    source_features_clip = []
    target_features_clip = []
    source_features_dino = []
    target_features_dino = []

    with torch.no_grad():
        print("Calculating features for source and target images")
        for i in range(0, len(source_image_paths), args.batch_size):
            print(f"Processing batch {i//args.batch_size + 1}/{len(source_image_paths)//args.batch_size + 1}")
            source_batch = source_image_paths[i : min(len(source_image_paths), i + args.batch_size)]
            target_batch = target_image_paths[i : min(len(target_image_paths), i + args.batch_size)]

            source_images = [download_image(path) for path in source_batch]
            target_images = [download_image(path) for path in target_batch]

            source_inputs_clip = clip_processor(text=None, images=source_images, return_tensors="pt", padding=True).to(device)
            target_inputs_clip = clip_processor(text=None, images=target_images, return_tensors="pt", padding=True).to(device)
            source_features_clip_batch = clip_model.get_image_features(**source_inputs_clip)  # (B, 768)
            target_features_clip_batch = clip_model.get_image_features(**target_inputs_clip)

            source_inputs_dino = dino_processor(source_images, return_tensors="pt", padding=True).to(device)
            target_inputs_dino = dino_processor(target_images, return_tensors="pt", padding=True).to(device)
            source_features_dino_batch = dino_model(**source_inputs_dino).last_hidden_state.mean(dim=1)
            target_features_dino_batch = dino_model(**target_inputs_dino).last_hidden_state.mean(dim=1)
            source_features_dino_batch = torch.nn.functional.normalize(source_features_dino_batch, p=2, dim=1)
            target_features_dino_batch = torch.nn.functional.normalize(target_features_dino_batch, p=2, dim=1)

            source_features_clip.append(source_features_clip_batch)
            target_features_clip.append(target_features_clip_batch)
            source_features_dino.append(source_features_dino_batch)
            target_features_dino.append(target_features_dino_batch)

    source_features_clip = torch.cat(source_features_clip, dim=0)  # (N, 768) [tensor]
    target_features_clip = torch.cat(target_features_clip, dim=0)  # (N, 768) [tensor]
    source_features_dino = torch.cat(source_features_dino, dim=0)  # (N, 768) [tensor]
    target_features_dino = torch.cat(target_features_dino, dim=0)  # (N, 768) [tensor]

    # Calculate clip similarity between all pairs of source and target images``
    # input source_features_clip (n*768) and target_features_clip (n*768)
    # output should be n*n matrix where (i,j) is the similarity between ith source and jth image
    source_target_clip_similarities_all = torch.zeros(len(source_image_paths), len(target_image_paths))
    with torch.no_grad():
        print("Calculating clip similarity between all pairs of source and target images")
        for i in range(0, len(source_image_paths), args.batch_size):
            source_batch = source_features_clip[i : min(len(source_image_paths), i + args.batch_size)]
            target_batch = target_features_clip
            similarities = F.cosine_similarity(source_batch.unsqueeze(1), target_batch.unsqueeze(0), dim=2).cpu()
            source_target_clip_similarities_all[i : min(len(source_image_paths), i + args.batch_size)] = similarities

    # Calculate clip similarity between all pairs of source and target images``
    # input source_features_dino (n*768) and target_features_dino (n*768)
    # output should be n*n matrix where (i,j) is the similarity between ith source and jth image
    source_target_dino_similarities_all = torch.zeros(len(source_image_paths), len(target_image_paths))
    with torch.no_grad():
        print("Calculating dino similarity between all pairs of source and target images")
        for i in range(0, len(source_image_paths), args.batch_size):
            source_batch = source_features_dino[i : min(len(source_image_paths), i + args.batch_size)]
            target_batch = target_features_dino
            similarities = F.cosine_similarity(source_batch.unsqueeze(1), target_batch.unsqueeze(0), dim=2).cpu()
            source_target_dino_similarities_all[i : min(len(source_image_paths), i + args.batch_size)] = similarities

    # combine clip and dino similarities with weighted sum
    print("Combining clip and dino similarities with weighted sum")
    source_target_similarities_all = args.clip_weight * source_target_clip_similarities_all + args.dino_weight * source_target_dino_similarities_all
    source_target_similarities_all = source_target_similarities_all.tolist()

    # find closest target image for each source image and write to file
    print("Finding closest target image for each source image and writing to file")
    with open(os.path.join(config["output_dir"], "closest_target_image.txt"), "w") as f:
        for i in range(len(source_image_paths)):
            max_sim = max(source_target_similarities_all[i])
            max_sim_idx = source_target_similarities_all[i].index(max_sim)
            f.write(f"{source_image_paths[i]}\t{target_image_paths[max_sim_idx]}\t{max_sim}\n")


if __name__ == "__main__":
    main()
