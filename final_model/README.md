# vlr-project: Focus on one-step diffusion models for image-editing
Setup the environment with `environment.yaml` 
```
conda env create -f environment.yaml
```

- The training scripts expect the dataset to be in the following format:





    ```
    data
    ├── dataset_name
    │   ├── train_A
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── train_B
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── fixed_prompt_a.txt
    |   └── fixed_prompt_b.txt
    |   └── targets_a.txt
    |   └── targets_b.txt
    |   └── captions_a.txt
    |   └── captions_b.txt
    |
    |   ├── test_A
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── test_B
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    ```

where `targets` files are used for supervised training and `captions` files are use for specific original and edited captions for each image.



You can launch the training of our enhanced CycleGAN-turbo with the command: 

```
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/final_sup" \
    --dataset_folder "data/your_data" \
    --train_img_prep "resize_286_randomcrop_256x256_hflip" --val_img_prep "resize_286_randomcrop_256x256_hflip" \
    --learning_rate="5e-4" --max_train_steps=25000 \
    --train_batch_size=1 --gradient_accumulation_steps=3 \
    --tracker_project_name "sup_final" \
    --enable_xformers_memory_efficient_attention --validation_steps 250 \
    --lambda_gan 1 --lambda_idt 0.2 --lambda_cycle 0.01 --lambda_cycle_lpips 0.2 --lambda_idt_lpips 0.3 --lambda_sup 0.7 --lambda_sup_lpips 1  --report_to wandb
```