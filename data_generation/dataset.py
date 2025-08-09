import os
import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
from daam.utils import compute_token_merge_indices


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, json_file_name, transform, tokenizer):
        """
        Parameters:
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): the image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.data_json = os.path.join(dataset_folder, json_file_name)
        with open(self.data_json, "r") as f:
            data_dict = json.load(f)
            self.data = list(data_dict.items())
            f.close()
        
        self.tokenizer = tokenizer
        
        self.T = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values": The processed image
            - "input_ids": The domain's fixed caption tokenized.
        """
        img_path, prompt = self.data[index]
        
        input_id = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]

        img_pil = Image.open(os.path.join(self.dataset_folder, img_path)).convert("RGB")
        img_t = F.to_tensor(self.T(img_pil))
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        return {
            "pixel_values": img_t,
            "input_ids": input_id,
        }
    


class TokenDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(self, dataset_folder, json_file_name, transform, tokenizer, split, word_tokens=None, new_tokens=None):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.data_json = os.path.join(dataset_folder, json_file_name)
        with open(self.data_json, "r") as f:
            data_dict = json.load(f)
            self.data = list(data_dict.items())
            f.close()
        
        self.tokenizer = tokenizer
        self.word_tokens = word_tokens
        self.new_tokens = new_tokens
        
        self.T = transform
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, prompt = self.data[index]

        new_tokens_start = []
        if self.word_tokens is not None and self.new_tokens is not None:
            for i in range(len(self.word_tokens)): 
                if self.word_tokens[i] in prompt:
                    first_word = self.word_tokens[i].split(" ")[0]  # there might be multiple words 
                    word_idx, _ = compute_token_merge_indices(self.tokenizer, prompt, first_word)
                    new_tokens_start += word_idx
                    prompt = prompt.replace(self.word_tokens[i], self.new_tokens[i] + " " + self.word_tokens[i])
                else:
                    new_tokens_start.append(-1)

        img_pil = Image.open(os.path.join(self.dataset_folder, img_path)).convert("RGB")
        img_t = self.T(img_pil)
        new_tokens_start = torch.tensor(new_tokens_start)

        input_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]

        return {
            "input_ids": input_ids,
            "pixel_values": img_t,
            "new_tokens_start": new_tokens_start,
        }