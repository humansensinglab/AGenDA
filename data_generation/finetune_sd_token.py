import itertools
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utils_attndb import *
from hook import UNetCrossAttentionHooker
from dataset import TokenDataset
from attndb_clip import CLIPTextModel


if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def compute_snr(noise_scheduler, timesteps):  # copy from higher version diffusion
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    for the given timesteps using the provided noise scheduler.

    Args:
        noise_scheduler (`NoiseScheduler`):
            An object containing the noise schedule parameters, specifically `alphas_cumprod`, which is used to compute
            the SNR values.
        timesteps (`torch.Tensor`):
            A tensor of timesteps for which the SNR is computed.

    Returns:
        `torch.Tensor`: A tensor containing the computed SNR values for each timestep.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, validation_prompts, object_tokens, step):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
        
    image_logs = []
    for prompt in validation_prompts:
        images = []
        object_token_prompt = []
        for init_token, new_token in zip(args.initialize_token, object_tokens):
            if init_token in prompt:
                object_token_prompt.append(new_token)
        
        prompt = prompt.format(*object_token_prompt)
        for i in range(4):
            generator = torch.Generator(device=accelerator.device).manual_seed(i)

            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(accelerator.device.type)

            with autocast_ctx:
                image = pipeline(prompt, num_inference_steps=20, generator=generator).images[0]
                images.append(image)

        image_logs.append(
            {"images": images, "prompt": prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                formatted_images = []
                
                for image in images:
                    formatted_images.append(np.asarray(image))
                
                formatted_images = np.stack(formatted_images)
                validation_prompt = log["prompt"]
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            for log in image_logs:
                tracker.log(
                    {
                        f"validation/{log['prompt']}": [
                            wandb.Image(image, caption=log["prompt"])
                            for i, image in enumerate(log["images"])
                        ]
                    }
                )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def save_learned_embedding(object_tokens, training_embedding, save_path):
    logger.info("Saving embeddings")
    learned_embeds_dict = dict()
    for token, embed in zip(object_tokens, training_embedding):
        learned_embeds_dict[token] = embed.detach().cpu()
    torch.save(learned_embeds_dict, save_path)


def save_full_model(accelerator, unet, output_dir, skip_save_unet=False):
    pipeline_args = {}
    pipeline_args["unet"] = accelerator.unwrap_model(unet) if not skip_save_unet else None

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        **pipeline_args,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline.save_pretrained(output_dir)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=None,
        help=(
            "A folder containing the source data."
        ),
    )
    parser.add_argument(
        "--json_file_name",
        type=str,
        default=None,
        help=(
            "The JSON file name that contains the image paths and prompts."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default)'
            ', `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompts`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompts` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--skip_save_text_encoder", action="store_true", required=False, help="Set to not save text encoder"
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="The path to the text embeddings learned in the first stage.",
    )
    parser.add_argument(
        "--train_token",
        action="store_true",
        required=False,
        default=False,
        help="Whether to train Textual Inversion.",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        required=False,
        default=False,
        help="Whether to train U-Net.",
    )
    parser.add_argument(
        "--object_token",
        type=str,
        default='sks',
        help="The token that represent the target concept. We will add `_v{n}` at the end of the token depends on `--n_object_embedding`",
    )
    parser.add_argument(
        "--n_object_embedding",
        type=int,
        default=1,
        help="The number of text embeddings that represent the target concept.",
    )
    parser.add_argument(
        "--initialize_token",
        type=str,
        default=None,
        nargs="+",
        help="The initialization for `[V]` in the first stage. Please remain consistent with `--n_object_embedding`, otherwise we may truncate it to `--n_object_embedding`",
    )
    parser.add_argument(
        "--train_cross_attn",
        action="store_true",
        default=False,
        help="Whether to train the cross attention layer.",
    )
    parser.add_argument(
        "--with_cross_attn_reg",
        default=False,
        action="store_true",
        help="Whether to use cross attention regularization.",
    )
    parser.add_argument(
        "--reg_weight", 
        type=float, 
        default=1.0, 
        help="The weight of the cross attention regularization."
    )
    parser.add_argument(
        "--only_save_checkpoint",
        action="store_true",
        default=False,
        help="Whether to save the full model."
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="The path to the checkpoint to be loaded."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tensorboard",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    if args.dataset_folder is None or args.json_file_name is None:
        raise ValueError("Need either a dataset name or a data json file.")
    
    if not (args.train_token or args.train_unet or args.train_cross_attn):
        raise ValueError("choose something to train! `--train_token`, `--train_cross_attn` or `--train_unet`")
    
    if args.train_unet and args.train_cross_attn:
        raise ValueError("`--train_unet` cannot be used with `--train_cross_attn`")

    if len(args.initialize_token)==0 and not args.embedding_path:
        raise ValueError("You must specify at least one token for initialization.")
    
    if args.load_from_checkpoint is not None and args.resume_from_checkpoint is not None:
        raise ValueError("`--load_from_checkpoint` cannot be used with `--resume_from_checkpoint`")

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    if args.train_token and (args.train_unet or args.train_cross_attn) and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training TI in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    if args.train_token:
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
    else:
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )

    if args.embedding_path is not None:
        embeds_dict=torch.load(args.embedding_path)
        object_tokens=list(embeds_dict.keys())
        initialize_embeds =torch.stack([embeds_dict[token]for token in object_tokens])
        num_new_tokens = tokenizer.add_tokens(object_tokens)
        object_token_ids = tokenizer.convert_tokens_to_ids(object_tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            text_encoder.get_input_embeddings().weight.data[object_token_ids] = initialize_embeds


    else:
        object_tokens=[args.object_token+f'_v{i}' for i in range(len(args.initialize_token))]

        num_new_tokens = tokenizer.add_tokens(object_tokens)
        object_token_ids = tokenizer.convert_tokens_to_ids(object_tokens)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        initialize_embeds= []       
        if len(args.initialize_token)>0:
            for init_token in args.initialize_token:
                init_token_embed = torch.mean(get_token_embeds(init_token, tokenizer, text_encoder),dim=0, keepdim=True)
                initialize_embeds.append(init_token_embed)
        
        initialize_embeds = torch.concatenate(initialize_embeds, dim=0)
        initialize_embeds = initialize_embeds.to(accelerator.device)

        training_embedding = torch.empty_like(initialize_embeds)
        training_embedding = nn.Parameter(training_embedding)
        nn.init.normal_(training_embedding, std=0.02)


    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    # Load in the weights from the second stage
    if args.load_from_checkpoint:
        path = Path(args.load_from_checkpoint)

        if not path.exists():
            accelerator.print(
                f"Checkpoint '{args.load_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.load_from_checkpoint = None
            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
            )
        else:
            accelerator.print(f"Loading from checkpoint {path.name}")
            unet = UNet2DConditionModel.from_pretrained(
                path, subfolder="unet", revision=args.revision
            )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if not args.train_unet:
        unet.requires_grad_(False)
        unet.eval()
    
    if args.train_cross_attn:
        unfreeze_model(unet, ['attn2'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if args.with_cross_attn_reg:
        attn_proc_hooker=UNetCrossAttentionHooker(True)
        original_attn_proc=unet.attn_processors
        unet.set_attn_processor(attn_proc_hooker)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize=[]
    if args.train_token:
        params_to_optimize += [[training_embedding]]
    if args.train_unet:
        params_to_optimize += [unet.parameters()]
    if args.train_cross_attn and not args.train_unet:
        params_to_optimize += [filter(lambda p:p.requires_grad,unet.parameters())]
    params_to_optimize = itertools.chain(*params_to_optimize)

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
        
    # Dataset and DataLoaders creation:
    train_dataset = TokenDataset(
        dataset_folder=args.dataset_folder,
        transform=image_transforms,
        tokenizer=tokenizer,
        json_file_name=args.json_file_name,
        split="train",
        word_tokens = args.initialize_token,
        new_tokens=object_tokens,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    modules_to_prepare=[optimizer, train_dataloader, lr_scheduler]
    if args.train_token:
        modules_to_prepare.append(training_embedding)
    if args.train_unet or args.train_cross_attn:
        modules_to_prepare.append(unet)

    # Prepare everything with our `accelerator`.
    if args.train_token and (args.train_unet or args.train_cross_attn):
        optimizer, train_dataloader, lr_scheduler, training_embedding, unet = accelerator.prepare(
            *modules_to_prepare
        )
    elif args.train_token:
        optimizer, train_dataloader, lr_scheduler, training_embedding = accelerator.prepare(
            *modules_to_prepare
        )
    else:
        optimizer, train_dataloader, lr_scheduler, unet = accelerator.prepare(
            *modules_to_prepare
        )

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.train_unet and not args.train_cross_attn:
        unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("initialize_token")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    if accelerator.sync_gradients and accelerator.is_main_process and args.validation_prompts is not None:
        if args.with_cross_attn_reg:
            attn_proc_hooker.is_train=False
        
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet if not args.train_unet and not args.train_cross_attn else accelerator.unwrap_model(unet),
            args,
            accelerator,
            weight_dtype,
            args.validation_prompts,
            object_tokens,
            global_step,
        )

        if args.with_cross_attn_reg:
            attn_proc_hooker.is_train=True
            attn_proc_hooker.clear()

    if args.train_unet or args.train_cross_attn:
        unet.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            if args.train_unet or args.train_cross_attn:
                accumulate_model=unet
            else:
                accumulate_model=training_embedding

            with accelerator.accumulate(accumulate_model):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Sample noise that we'll add to the model input
                if args.offset_noise:
                    noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                        model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                    )
                else:
                    noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                new_tokens_start_indices = batch["new_tokens_start"]

                # Get the text embedding for conditioning
                if args.train_token:
                    batch_embeddings = text_encoder.get_input_embeddings().weight.data[batch["input_ids"]]
                    for i in range(bsz):
                        for start_index, single_word_embedding in zip(new_tokens_start_indices[i], training_embedding):
                            if start_index > 0:
                                batch_embeddings[i][start_index:start_index+args.n_object_embedding] = single_word_embedding

                if args.train_token:
                    encoder_hidden_states = text_encoder(batch["input_ids"], inputs_embeds=batch_embeddings)[0]
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                # Clear the attention maps
                if args.with_cross_attn_reg:
                    attn_proc_hooker.clear()    

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels, return_dict=False
                )[0]

                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # cross attn loss
                attn_loss=torch.tensor(0.0).to(accelerator.device)
                bg_attn_loss = torch.tensor(0.0).to(accelerator.device)
                fg_attn_loss = torch.tensor(0.0).to(accelerator.device)
                if args.with_cross_attn_reg:
                    cross_attn_maps=attn_proc_hooker.cross_attn_maps
                    for attn_maps in cross_attn_maps:  # (batch_size, tokens, num_pixels)
                        for sample_attn_map, sample_start_indices in zip(attn_maps, new_tokens_start_indices):
                            if sample_start_indices[0] > 0: # whethere there is a car in the image
                                obj_index = sample_start_indices[0]+args.n_object_embedding
                                obj_attn_map = sample_attn_map[obj_index]
                                norm_obj_attn_map = (obj_attn_map - obj_attn_map.min()) / (obj_attn_map.max() - obj_attn_map.min() + 1e-8)
                                norm_bg_ref_attn_map = 1 - norm_obj_attn_map
                                norm_bg_ref_attn_map = norm_bg_ref_attn_map / torch.sum(norm_bg_ref_attn_map)
                                norm_obj_attn_map = norm_obj_attn_map / torch.sum(norm_obj_attn_map)
                                
                                fg_attn_map = sample_attn_map[sample_start_indices[0]]  # foreground token attention map loss
                                norm_fg_attn_map = (fg_attn_map - fg_attn_map.min()) / (fg_attn_map.max() - fg_attn_map.min() + 1e-8)
                                norm_fg_attn_map = norm_fg_attn_map / torch.sum(norm_fg_attn_map)

                                bg_index = sample_start_indices[sample_start_indices > -1][-1] # background token attention map loss
                                bg_attn_map = sample_attn_map[bg_index]
                                norm_bg_attn_map = (bg_attn_map - bg_attn_map.min()) / (bg_attn_map.max() - bg_attn_map.min() + 1e-8)
                                norm_bg_attn_map = norm_bg_attn_map / torch.sum(norm_bg_attn_map)

                                bg_attn_loss += args.reg_weight*torch.mean(torch.abs(norm_bg_ref_attn_map - norm_bg_attn_map)) / torch.sum(new_tokens_start_indices[:,0]>0)
                                fg_attn_loss += args.reg_weight*torch.mean(torch.abs(norm_obj_attn_map - norm_fg_attn_map)) / torch.sum(new_tokens_start_indices[:,0]>0)
                                attn_loss = bg_attn_loss + fg_attn_loss

                    attn_loss=attn_loss / len(cross_attn_maps)
                    attn_proc_hooker.clear()

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                loss += attn_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.train_token:
                    params_to_clip = itertools.chain(filter(lambda p:p.requires_grad,unet.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if args.with_cross_attn_reg:
                            accelerator.unwrap_model(unet).set_attn_processor(original_attn_proc)
                        accelerator.save_state(save_path)
                        if args.with_cross_attn_reg:
                            original_attn_proc=accelerator.unwrap_model(unet).attn_processors
                            accelerator.unwrap_model(unet).set_attn_processor(attn_proc_hooker)
                        logger.info(f"Saved state to {save_path}")                       
                        if args.train_token:
                            save_path=os.path.join(args.output_dir, f"checkpoint-{global_step}", f"learned_embeds_steps_{global_step}.bin")
                            save_learned_embedding(object_tokens, training_embedding, save_path)


                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        if args.train_token:
                            with torch.no_grad():
                                text_encoder.get_input_embeddings().weight.data[object_token_ids]=training_embedding

                        if args.with_cross_attn_reg:
                            attn_proc_hooker.is_train=False
                        
                        images = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet if not args.train_unet and not args.train_cross_attn else accelerator.unwrap_model(unet),
                            args,
                            accelerator,
                            weight_dtype,
                            args.validation_prompts,
                            object_tokens,
                            global_step,
                        )

                        if args.with_cross_attn_reg:
                            attn_proc_hooker.is_train=True
                            attn_proc_hooker.clear()

            logs = {
                "loss": loss.detach().item(),
                "attn_loss":attn_loss.detach().item(),
                "fg_loss": fg_attn_loss.detach().item(),
                'bg_loss': bg_attn_loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.train_token:
            save_path=os.path.join(args.output_dir,f"learned_embeds_steps_{global_step}.bin")
            save_learned_embedding(object_tokens, training_embedding, save_path)
        if not args.only_save_checkpoint and (args.train_unet or args.train_cross_attn):
            save_path=os.path.join(args.output_dir,f"full_model_step_{global_step}")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                if args.with_cross_attn_reg:
                    accelerator.unwrap_model(unet).set_attn_processor(original_attn_proc)
                save_full_model(accelerator, unet, save_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        setup_seed(args.seed)
    main(args)