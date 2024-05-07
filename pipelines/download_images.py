import csv
import pandas as pd
from image_retrieve import download_one_image
import json

metadata = "/home/skhanuja/vlr-project/closest_target_image_in-jp.csv"
captions = "/home/skhanuja/vlr-project/outputs/caption-llm_edit/japan-india/metadata.csv"

captions_df = pd.read_csv(captions)

filename2captions = {}
for index, row in captions_df.iterrows():
    filename = row["src_image_path"].split('/')[-1]
    caption_original = row["caption"]
    caption_edited = row["llm_edit"]
    filename2captions[filename] = caption_original

# Load the CSV file into a DataFrame
df = pd.read_csv(metadata)

# get src_path, tgt_path
src2tgt = {}
final_captions = {}
output_src_path = "/data/tir/projects/tir4/corpora/datacomp-1b/food/laion/high_quality/pix2pix_turbo_india-japan/train_A"
output_tgt_path = "/data/tir/projects/tir4/corpora/datacomp-1b/food/laion/high_quality/pix2pix_turbo_india-japan/train_B"
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Process each row as needed
    src_path = row["src_path"]
    tgt_path = row["tgt_path"]
    src2tgt[src_path] = tgt_path
    filename = f"{index}.jpg"
    tgt_basepath = tgt_path.split('/')[-1]
    caption = filename2captions[tgt_basepath]
    # download_one_image(src_path, output_src_path, filename)
    # download_one_image(tgt_path, output_tgt_path, filename)
    final_captions[filename] = caption

with open("/data/tir/projects/tir4/corpora/datacomp-1b/food/laion/high_quality/pix2pix_turbo_india-japan/train_prompts.json", "w") as json_file:
    json.dump(final_captions, json_file)





