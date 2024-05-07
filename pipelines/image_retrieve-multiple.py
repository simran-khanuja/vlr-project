import os
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions
import requests
from pathlib import Path
from PIL import Image
import json
import random
import logging
import argparse
import yaml
import pandas as pd
from PIL import Image, ImageOps
from transformers import ViTImageProcessor, ViTModel, CLIPModel, CLIPProcessor, CLIPTokenizer
import torch
import torch.nn.functional as F
from io import BytesIO
from collections import defaultdict
from aesthetics_predictor import AestheticsPredictorV2Linear
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_local_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def download_image(url):
    try:
        response = requests.get(url, timeout=10)  # Adjust timeout as needed
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = image.convert('RGB')  # Normalize the image format
            return image
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {str(e)}")
    return None

def download_images(src_image_paths, temp_path):
    os.makedirs(temp_path, exist_ok=True)
    # Use ThreadPoolExecutor to manage multiple download threads
    url_target_paths = []
    for src_image_path in src_image_paths:
        target_path = f"{temp_path}/{src_image_path.split('/')[-1][:-4]}"
        src_urls_all_cols = src_image_paths[src_image_path]
        for col in src_urls_all_cols:
            col_target_path = f"{target_path}/{col}"
            os.makedirs(col_target_path, exist_ok=True)
            for i, url in enumerate(src_image_paths[src_image_path][col]):
                url_target_paths.append({"url": url, "target_path": f"{col_target_path}/{i}.jpg"})


    # Download images in parallel but need to control RAM usage
    batch_size = 1000
    results = []
    # Process in batches
    for i in range(0, len(url_target_paths), batch_size):
        batch = url_target_paths[i:i+batch_size] if i+batch_size < len(url_target_paths) else url_target_paths[i:]
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(download_image, entry["url"]): entry for entry in batch}
            batch_results = []
            for future in as_completed(future_to_url):
                entry = future_to_url[future]
                try:
                    image = future.result()
                    if image is not None:
                        # save image
                        image.save(entry["target_path"])
                        batch_results.append(entry)
                except Exception as e:
                    logging.error(f"Image download failed for {entry['url']}: {str(e)}")
            results.extend(batch_results)
    return results
    
def get_best_image(target_path, source_image, output_folder, dino_model, dino_processor, clip_model, clip_processor, aes_predictor, aes_processor):
    batch_size = 8

    # get images from target path
    urls = [f"{target_path}/{f}" for f in os.listdir(target_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

    downloaded_src_image = download_local_image(source_image)
        
    # Get dino similarity scores for all images
    cosine_dino = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size] if i+batch_size < len(urls) else urls[i:]
        source_images = [downloaded_src_image] * len(batch_urls)
        target_images = [download_local_image(url) for url in batch_urls]
        source_inputs = dino_processor(source_images, return_tensors="pt", padding=True).to("cuda")
        target_inputs = dino_processor(target_images, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            source_features = dino_model(**source_inputs).last_hidden_state.mean(dim=1)
            target_features = dino_model(**target_inputs).last_hidden_state.mean(dim=1)
            source_features = torch.nn.functional.normalize(source_features, p=2, dim=1)
            target_features = torch.nn.functional.normalize(target_features, p=2, dim=1)
            cosine_dino.extend(F.cosine_similarity(source_features, target_features).cpu().tolist())
    
    # Get CLIP similarity scores for all images
    cosine_clip = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size] if i+batch_size < len(urls) else urls[i:]
        source_images = [downloaded_src_image] * len(batch_urls)
        target_images = [download_local_image(url) for url in batch_urls]
        source_inputs = clip_processor(images=source_images, return_tensors="pt", padding=True).to("cuda")
        target_inputs = clip_processor(images=target_images, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            source_features = clip_model.get_image_features(**source_inputs)
            target_features = clip_model.get_image_features(**target_inputs)
            source_features = torch.nn.functional.normalize(source_features, p=2, dim=1)
            target_features = torch.nn.functional.normalize(target_features, p=2, dim=1)
            cosine_clip.extend(F.cosine_similarity(source_features, target_features).cpu().tolist())
    
    # get aesthetics scores
    aesthetics_scores = []
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size] if i+batch_size < len(urls) else urls[i:]
        target_images = [download_local_image(url) for url in batch_urls]
        inputs = aes_processor(images=target_images, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = aes_predictor(**inputs)
        aesthetics_scores.extend(outputs.logits.cpu().view(-1).tolist())
    
    # Get the best image based max clip and dino similarity (0.5 wt to each)
    best_image = None
    best_score = -1
    for i in range(len(urls)):
        score = 0.5 * cosine_dino[i] + 0.5 * cosine_clip[i]
        if score > best_score and aesthetics_scores[i] > 4.5:
            best_score = score
            best_image = urls[i]
    
    # Copy the best image to the output folder
    os.makedirs(output_folder, exist_ok=True)
    col = target_path.split("/")[-1]
    if best_image is not None:
        shutil.copy(best_image, f"{output_folder}/{source_image.split('/')[-1][:-4]}_{col}.jpg")
        return f"{output_folder}/{source_image.split('/')[-1][:-4]}_{col}.jpg"
    return None


def main():
    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

    # read in the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/skhanuja/vlr-project/configs/cap-retrieve/india.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read in the metadata file
    data = pd.read_csv(config["metadata_path"])

    # take 10 rows in data
    # data = data.head(10)

    src_image_paths = data[config["src_image_path_col"]].tolist()
    blip_captions = data[config["caption_col"]].tolist()
    llm_edit_cols = [f"llm_edit{i}" for i in range(1, 6)]
    llm_edits = {}
    for col in llm_edit_cols:
        llm_edits[col] = data[col].tolist()
    
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # setup dino model
    dino_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
    dino_model = ViTModel.from_pretrained('facebook/dino-vitb8').eval().to(device)

    # setup aesthetic model
    # Load CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # aesthetics predictor
    model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
    aes_predictor = AestheticsPredictorV2Linear.from_pretrained(model_id).to("cuda")
    aes_processor = CLIPProcessor.from_pretrained(model_id)
    
    # setup clip options
    clip_options = ClipOptions(
        indice_folder = config["indice_folder"],
        clip_model = config["clip_model"],
        enable_hdf5 = False,
        enable_faiss_memory_mapping = True,
        columns_to_return = ["image_path", "caption", "url", "original_height", "original_width"],
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

    captions = {}
    image_urls = {}
    for i, src_image_path in enumerate(src_image_paths):
        prompts = {}
        for col in llm_edit_cols:
            prompts[col] = llm_edits[col][i]
        # get basename of the image and append country to the basename
        captions[src_image_path] = defaultdict(list)
        image_urls[src_image_path] = defaultdict(list)
        
        for col in prompts:
            results = knn_service.query(text_input=prompts[col],
                                        modality="image", 
                                        indice_name=config["indice_name"], 
                                        num_images=100, 
                                        num_result_ids=100,
                                        deduplicate=True)
        
            for result in results:
                if result['original_height'] < 256 or result['original_width'] < 256:
                    continue
                captions[src_image_path][col].append(result["caption"])
                image_urls[src_image_path][col].append(result["url"])
        
    # try parallel download of all images to a temp folder
    temp_folder = config["temp_path"]
    results = download_images(image_urls, temp_folder)

    metadata_file = open(config["output_file"], "w")
    # find the best image for each prompt
    for i, src_image_path in enumerate(src_image_paths):
        for col in llm_edit_cols:
            target_folder = f"{temp_folder}/{src_image_path.split('/')[-1][:-4]}/{col}"
            tgt_path = get_best_image(target_folder, src_image_path, config["tgt_image_path"], dino_model, dino_processor, clip_model, clip_processor, aes_predictor, aes_processor)
            if tgt_path is not None:
                metadata_file.write(f"{src_image_path},{col},{tgt_path}\n")
            
    metadata_file.close()

    
    


            

    
if __name__ == "__main__":
    main()



