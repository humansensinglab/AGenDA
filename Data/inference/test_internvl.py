import os
import argparse
from PIL import Image
import json
from tqdm import tqdm

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Example of testing on UGRC using InternVL3.")
    parser.add_argument(
        "--test_data_base_path", type=str, default="Data/Real/UGRC/test", help="test data folder path",
    )
    parser.add_argument(
        "--annotation_file", type=str, default="annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json", help="ground truth coco annotation",
    )
    parser.add_argument(
        "--save_path", type=str, default="annotations_internvl.json", help="save path for predicted annotations",
    )
    parser.add_argument(
        "--image_width", type=int, default=112, help="image width",
    )
    parser.add_argument(
        "--image_height", type=int, default=112, help="image height",
    )
    args = parser.parse_args()
    return args


def parse_list_boxes(text):
    result = []
    for line in text.strip().splitlines():
        # Extract the numbers from the line, remove brackets and split by comma
        try:
            numbers = line.split('[')[1].split(']')[0].split(',')
            result.append([int(num.strip()) for num in numbers])
        except:
            continue

    return result


if __name__ == "__main__":
    args = parse_args()

    model = 'OpenGVLab/InternVL3-8B'
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
    
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

    prompt = "Please provide the bounding box coordinate of all cars in the image using the format [x1, y1, x2, y2]."

    for image_ann in tqdm(image_list):
        img = Image.open(os.path.join(test_data_base_path, "images", image_ann["file_name"]))

        response = pipe((prompt, img))

        results = parse_list_boxes(response.text)

        if len(results) > 0:
            for bbox in results:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if y1 > y2:
                        y1, y2 = y2, y1
                    if x1 > x2:
                        x1, x2 = x2, x1
                    y1, x1, y2, x2 = y1 / 1000* args.image_height, x1 / 1000* args.image_width, y2 / 1000* args.image_height, x2 / 1000* args.image_width
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