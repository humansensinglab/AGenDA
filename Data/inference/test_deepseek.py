import os
import argparse
import re
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


def parse_list_boxes(text):
    result = []
    try:
        bbox_strings = re.findall(r'\[(\d+), (\d+), (\d+), (\d+)\]', text)
        result = [list(map(int, bbox)) for bbox in bbox_strings]
    except:
        result = []

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Example of testing on UGRC using Deepseek-VL2.")
    parser.add_argument(
        "--test_data_base_path", type=str, default="Data/Real/UGRC/test", help="test data folder path",
    )
    parser.add_argument(
        "--annotation_file", type=str, default="annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json", help="ground truth coco annotation",
    )
    parser.add_argument(
        "--save_path", type=str, default="annotations_deepseek.json", help="save path for predicted annotations",
    )
    parser.add_argument(
        "--image_width", type=int, default=112, help="image width",
    )
    parser.add_argument(
        "--image_height", type=int, default=112, help="image height",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # image and folder path
    test_data_base_path = args.test_data_base_path
    coco_filename = args.annotation_file
    pred_coco_save_path = args.save_path
    coco_pred_ann = {}

    with open(os.path.join(test_data_base_path, coco_filename)) as f:
        coco_gt_ann = json.load(f)

    coco_pred_ann["categories"] = coco_gt_ann["categories"].copy()
    coco_pred_ann['images'] = coco_gt_ann['images'].copy()
    coco_pred_ann['annotations'] = []
    image_list = coco_gt_ann["images"]
    object_id = 0

    for image_ann in tqdm(image_list):
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n<|ref|>Cars.<|/ref|>.",
                "images": [f"{test_data_base_path}/images/{image_ann['file_name']}"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        results = parse_list_boxes(answer)

        if len(results) > 0:
            for bbox in results:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if y1 > y2:
                        y1, y2 = y2, y1
                    if x1 > x2:
                        x1, x2 = x2, x1
                    y1, x1, y2, x2 = y1 / 999* args.image_height, x1 / 999* args.image_width, y2 / 999* args.image_height, x2 / 999* args.image_width
                    w, h = x2 - x1, y2 - y1
                    new_ann = {}
                    new_ann["iscrowd"] = 0
                    new_ann["category_id"] = 1
                    new_ann["id"] = object_id
                    new_ann['image_id'] = image_ann['id']
                    new_ann['bbox'] = [x1, y1, w, h]
                    new_ann['area'] = w*h
                    coco_pred_ann["annotations"].append(new_ann)

                    object_id +=1


    with open(pred_coco_save_path, "w") as f:
        json.dump(coco_pred_ann,f,indent=4)