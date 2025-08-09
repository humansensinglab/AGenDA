import os
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import daam
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Image and attention map generation.")
    parser.add_argument("--save-dir", type=str, default="Data/Synthetic", help="Directory to save images (and heatmaps if enabled).")
    parser.add_argument("--pretrained-model-path", type=str, default = "output/LINZ-Utah/sd1.4-token-finetune-stage-two/full_model_step_4500", help="Path or repo-id of the pretrained model to load.")
    parser.add_argument("--learnable-tokens-embedding-path", type=str, default = "output/LINZ-Utah/sd1.4-token-finetune-stage-one/learned_embeds_steps_9000.bin", help="Path to the learned token embeddings.")
    parser.add_argument("--prompt", type=str, default="An aerial view image with {} cars in {} Utah", help="Prompt template for image generation.")
    parser.add_argument("--initialize_token", type=str, default = ["cars", "Utah", "New Zealand"], nargs="+", help="The initialization for learnable tokens in the first stage")
    parser.add_argument("--word_token_heatmaps", type=str, default=None, nargs="+", help="word tokens to compute DAAM heatmaps.")
    parser.add_argument("--store_learnable_token_heatmaps", action="store_true", help="Whether to store DAAM heatmaps for learnable tokens.")
    parser.add_argument("--num-images", type=int, default=10000, help="Number of images to generate.")
    parser.add_argument("--image-size", type=int, default=112, help="Size of the generated images.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_path)
    pipeline = pipeline.to("cuda")

    embeds_dict=torch.load(args.learnable_tokens_embedding_path)
    all_new_tokens = list(embeds_dict.keys())

    all_word_token_heatmaps = args.word_token_heatmaps if args.word_token_heatmaps is not None else []
    new_tokens = []
    learnable_tokens_embeds = []
    for t, n in zip(args.initialize_token, all_new_tokens):
        if t in args.prompt:
            if args.store_learnable_token_heatmaps:
                all_word_token_heatmaps.append(n)
            new_tokens.append(n)

    learnable_tokens_embeds = torch.stack([embeds_dict[token] for token in new_tokens]).to("cuda")

    num_new_tokens = pipeline.tokenizer.add_tokens(new_tokens)
    new_token_ids = pipeline.tokenizer.convert_tokens_to_ids(new_tokens)
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))

    with torch.no_grad():
        pipeline.text_encoder.get_input_embeddings().weight.data[new_token_ids]= learnable_tokens_embeds

    prompt = args.prompt.format(*new_tokens)

    for seed in tqdm(range(args.num_images)):
        with daam.trace(pipeline) as daam_trc:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            output_image = pipeline(prompt, num_inference_steps=20, generator=generator).images[0]
            output_image = output_image.resize((args.image_size, args.image_size))
            if np.max(np.asarray(output_image))<1e-5:  # NSFW content filter
                continue

            daam_heatmap = daam_trc.compute_global_heat_map()
        
        if not os.path.exists(os.path.join(args.save_dir, "images")):
            os.makedirs(os.path.join(args.save_dir, "images"))
        output_image.save(os.path.join(args.save_dir, "images", f"{seed}.png"))

        for word in all_word_token_heatmaps:  # store DAAM heatmaps for all object tokens and learnable tokens
            if not os.path.exists(os.path.join(args.save_dir, "daam_"+word+"_heatmaps")):
                os.makedirs(os.path.join(args.save_dir, "daam_"+word+"_heatmaps"))

            object_word_heatmap = daam_heatmap.compute_word_heat_map(word)

            # Get the daam heatmap
            object_daam_heatmap = object_word_heatmap.heatmap
            object_daam_heatmap = object_daam_heatmap.cpu().detach().squeeze()
            object_daam_heatmap = object_daam_heatmap.numpy()

            # normalize heat map
            object_daam_heatmap = (object_daam_heatmap - object_daam_heatmap.min()) / (object_daam_heatmap.max() - object_daam_heatmap.min() + 1e-8) * 255

            word_DAAM_heat_pil = Image.fromarray(object_daam_heatmap.astype(np.uint8))
            word_DAAM_heat_pil = word_DAAM_heat_pil.resize((args.image_size, args.image_size))
            word_DAAM_heat_pil.save(os.path.join(args.save_dir, "daam_"+word+"_heatmaps", f"{seed}.png"))
