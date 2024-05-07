import os
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions
import requests
from pathlib import Path
import json
import random
import shutil
import logging
import argparse
import yaml
import pandas as pd
from aesthetics_predictor import AestheticsPredictorV2Linear
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor
    
    


def download_image(image_url, folder_path, processor, predictor, file_name=None):
    """
    Downloads an image from a given URL and saves it to a specified folder.

    :param image_url: URL of the image to download
    :param folder_path: Path to the folder where the image will be saved
    :param file_name: Name of the file (optional). If not provided, it will be derived from the URL.
    """
    # If file_name is not specified, extract it from the URL
    try:
        if not file_name:
            file_name = image_url.split('/')[-1]

        # Ensure folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # Get the image content
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # check aesthetic score of image
        image = Image.open(BytesIO(response.content))
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
            outputs = predictor(**inputs)
        prediction = outputs.logits.item()
        
        if prediction < 4.5:
            logging.info(f"Image aesthetic score is less than 4.5: {prediction}")
            return None
    
        # Save the image
        with open(Path(folder_path) / file_name, 'wb') as file:
            file.write(response.content)
        logging.info(f"Image saved as {Path(folder_path) / file_name}")

        file_path = Path(folder_path) / file_name
        file_size = os.stat(file_path).st_size

        if file_size == 0 or file_size < 1024:
            logging.info(f"Image saved is null or suspiciously small (size: {file_size} bytes)")
            return None
    
    except requests.RequestException as e:
        logging.info(f"Error downloading the image: {e}")
        return None
    except IOError as e:
        logging.info(f"Error saving the image: {e}")
        return None
    
    return "success"

def main():
    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # read in the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/retrieval_datacomp.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    prompt = config["prompt"]
        
    model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
    predictor = AestheticsPredictorV2Linear.from_pretrained(model_id).to("cuda")
    processor = CLIPProcessor.from_pretrained(model_id)


    # setup clip options
    clip_options = ClipOptions(
        indice_folder = config["indice_folder"],
        clip_model = config["clip_model"],
        enable_hdf5 = False,
        enable_faiss_memory_mapping = True,
        columns_to_return = ["image_path", "caption", "url", "original_height", "original_width", "clip_l14_similarity_score"],
        reorder_metadata_by_ivf_index = False,
        enable_mclip_option = False,
        use_jit = False,
        use_arrow = False,
        provide_safety_model = False,
        provide_violence_detector =  False,
        provide_aesthetic_embeddings =  False,
    )

    # load indices
    loaded_indices = load_clip_indices(config["indices_path"], clip_options)

    # construct the knn search object
    knn_service = KnnService(clip_resources=loaded_indices)

    captions = []
    image_paths = []
    image_urls = []
    ht_width = []
    clip_score = []
        
    results = knn_service.query(text_input=prompt,
                                modality="image", 
                                indice_name=config["indice_name"], 
                                num_images=10000, 
                                num_result_ids=10000,
                                deduplicate=True)
        
    count = 0
    for result in results:
        captions.append(result["caption"])
        image_paths.append(result["image_path"])
        image_urls.append(result["url"])
        ht_width.append((result["original_height"], result["original_width"]))
        clip_score.append(result["clip_l14_similarity_score"])
            
    logging.info("Retrieved images")

    random.seed(0)
    # make the output directory, first get directory name from output_file
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    for id, url in enumerate(image_urls):
        status = download_image(url, output_dir, processor, predictor, f"{id}.jpg")
        if status is None:
            logging.info(f"Failed to download image from {url}")
            continue

    logging.info("Done!")

if __name__ == "__main__":
    main()



