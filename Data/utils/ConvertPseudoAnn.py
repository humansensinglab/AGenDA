import json
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert predicted bboxes to pseudo annotations.")
    parser.add_argument(
        "--pred_file", type=str, help="predicted bbox file path",
    )
    parser.add_argument(
        "--pseudo_pred_file", type=str, help="pseudo annotation save path",
    )

    args = parser.parse_args()
    pred_file, pseudo_pred_file = args.pred_file, args.pseudo_pred_file

    with open(pred_file) as f:
        pred_ann_file = json.load(f)

    pseudo_pred_ann = {}
    pseudo_pred_ann['categories'] = pred_ann_file['categories']
    pseudo_pred_ann['images'] = pred_ann_file['images']
    pseudo_pred_ann['annotations'] = []

    bboxes_size_px = 42.36
    margin_px = bboxes_size_px/2-1
    image_size = (112, 112)

    for ann in tqdm(pred_ann_file['annotations']):
        l, t, w, h = ann['bbox']
        r, b = l+w, t+h
        x_c_bbox = (l+r)/2
        y_c_bbox = (t+b)/2
        
        if x_c_bbox < margin_px:
            r_full = r
            l_full = 0  
        elif x_c_bbox > image_size[0] - margin_px:
            l_full = l
            r_full = image_size[0]
        else:
            l_full = l
            r_full = r

        if y_c_bbox < margin_px:
            b_full = b
            t_full = 0
        elif y_c_bbox > image_size[1] - margin_px:
            t_full = t
            b_full = image_size[1]
        else:
            t_full = t
            b_full = b

        x_c_bbox_full = (l_full+r_full)/2
        y_c_bbox_full = (t_full+b_full)/2

        # Produce fake bbox annotations
        l = max(0, x_c_bbox_full-bboxes_size_px/2)
        t = max(0, y_c_bbox_full-bboxes_size_px/2)
        r = min(x_c_bbox_full+bboxes_size_px/2, image_size[0])
        b = min(y_c_bbox_full+bboxes_size_px/2, image_size[1])
        w = r - l
        h = b - t

        new_ann = ann.copy()
        new_ann['bbox'] = [l, t, w, h]
        new_ann['area'] = w*h
        new_ann['score'] = 1.0
        pseudo_pred_ann['annotations'].append(new_ann)

    with open(pseudo_pred_file,"w") as f:
        json.dump(pseudo_pred_ann, f, indent=4)