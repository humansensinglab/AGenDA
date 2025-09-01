# Data generation
This page introduces how to fine-tune Stable Diffusion and generate data.

## Data preparation
Throughout the process, We randomly sample a balanced subset with roughly equal numbers of images with cars and without cars from the LINZ and UGRC training sets. The image filenames and their corresponding template prompts are stored as key-value pairs in [train_data.json](train_data.json). The file train_data.json needs to be put under "<b>Data</b>" folder.

## Fine-tune Stable Diffusion
First, we fine-tune Stable Diffusion on both the LINZ and UGRC datasets to enhance its understanding of real-world aerial imagery.
```
bash data_generation/finetune_sd.sh
```

## Fine-tune learnable tokens
### First Stage
In this stage, we further fine-tune two learnable tokens to learn both the foreground (cars) and background (areas other than cars).
```
bash data_generation/finetune_sd_token.sh
```
The Stable Diffusion <b>checkpoint</b>  can be downloaded [here](https://huggingface.co/xiaofanghf/AGenDA-Finetune-Tokens-Stage1).

### Second Stage
In this stage, we fix the learned embeddings and further fine-tune Stable Diffusion.
```
bash data_generation/finetune_sd_token_stage2.sh
```
The Stable Diffusion <b>checkpoint</b>  can be downloaded [here](https://huggingface.co/xiaofanghf/AGenDA-Finetune-Tokens-Stage2).


## Synthetic data generation
We generate LINZ-style images with cars, UGRC-style images with cars, and UGRC-style images without cars.  

To generate LINZ-style images with cars:
```
python data_generation/data_generation.py \
    --pretrained-model-path output/LINZ-Utah/sd1.4-token-finetune-stage-two/full_model_step_4500 \
    --learnable-tokens-embedding-path output/LINZ-Utah/sd1.4-token-finetune-stage-one/learned_embeds_steps_9000.bin \
    --initialize_token "cars" "Utah" "New Zealand" \
    --save-dir Data/Synthetic/LINZ-with-cars \
    --prompt "An aerial view image with {} cars in {} New Zealand" \
    --word_token_heatmaps "cars" \
    --num-images 10000 \
    --store_learnable_token_heatmaps
```

To generate UGRC-style images with cars:
```
python data_generation/data_generation.py \
    --pretrained-model-path output/LINZ-Utah/sd1.4-token-finetune-stage-two/full_model_step_4500 \
    --learnable-tokens-embedding-path output/LINZ-Utah/sd1.4-token-finetune-stage-one/learned_embeds_steps_9000.bin \
    --initialize_token "cars" "Utah" "New Zealand" \
    --save-dir Data/Synthetic/UGRC-with-cars \
    --prompt "An aerial view image with {} cars in {} Utah" \
    --word_token_heatmaps "cars" \
    --num-images 10000 \
    --store_learnable_token_heatmaps
```

To generate UGRC-style images without cars:
```
python data_generation/data_generation.py \
    --pretrained-model-path output/LINZ-Utah/sd1.4-token-finetune-stage-two/full_model_step_4500 \
    --learnable-tokens-embedding-path output/LINZ-Utah/sd1.4-token-finetune-stage-one/learned_embeds_steps_9000.bin \
    --initialize_token "cars" "Utah" "New Zealand" \
    --save-dir Data/Synthetic/UGRC-with-cars \
    --prompt "An aerial view image in {} Utah" \
    --num-images 10000 
```

Then we can stack the "car" heatmap, foreground heatmap and background in synthetic LINZ and UGRC dataset by running the following commands:
```
python data_generation/postprocess_heatmap.py \
    --save-dir Data/Synthetic/UGRC-with-cars \
    --object-heatmap-path daam_cars_heatmaps \
    --fg-heatmap-path daam_new_token_v0_heatmaps \
    --bg-heatmap-path daam_new_token_v1_heatmaps \
    --stack-heatmap-save-path daam_stack_heatmaps \
    --inv-heatmap-save-path daam_new_token_v1_inv_heatmaps

python data_generation/postprocess_heatmap.py \
    --save-dir Data/Synthetic/LINZ-with-cars \
    --object-heatmap-path daam_cars_heatmaps \
    --fg-heatmap-path daam_new_token_v0_heatmaps \
    --bg-heatmap-path daam_new_token_v2_heatmaps \
    --stack-heatmap-save-path daam_stack_heatmaps \
    --inv-heatmap-save-path daam_new_token_v2_inv_heatmaps
```
You can obtain the synthetic data we generate following the data downloading process [here](../Data/README.md).

