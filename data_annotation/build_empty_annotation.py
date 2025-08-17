import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Image and attention map generation.")
    parser.add_argument("--image-dir", type=str, default="Data/Synthetic/LINZ-with-cars/images", help="Directory where images are stored.")
    parser.add_argument("--save-dir", type=str, default="Data/Synthetic/LINZ-with-cars/annotations_coco_Empty.json", help="Directory to save the COCO annotation file.")
    parser.add_argument("--coco-dir", type=str, default="Data/Real/LINZ/test/annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json", help="Path to the COCO annotation as an example.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    all_images = sorted(os.listdir(args.image_dir),key=lambda x:int(x.split(".")[0]))


    with open(args.coco_dir,"r") as f:
        all_ann_file = json.load(f)

    train_ann_dict = {}
    train_ann_dict['categories'] = all_ann_file['categories']
    item = all_ann_file['images'][0]
    # item.pop("seg_file_name")
    train_ann_dict['images'] = []
    train_ann_dict['annotations'] = []

    image_id = 0
    for name in all_images:
        item["id"] = image_id
        item['file_name'] = name
        item['height'] = 112
        item['width'] = 112
        image_id +=1
        train_ann_dict['images'].append(item.copy())

    with open(args.save_dir,"w") as f:
        json.dump(train_ann_dict, f, indent=4)