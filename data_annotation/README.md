# Data annotation
This page introduces how to automatically annotate synthetic UGRC images via cross-attention maps.

First, we train detectors on real source domain data.  We provide an example command using faster-rcnn, while other configuration files are put under the [configs](configs) folder.
```
cd mmdetection
# Train on real source domain training set
python tools/train.py ../data_annotation/configs/Real_Source/faster-rcnn.py

# Test on real source domain test set
python tools/test.py ../data_annotation/configs/Real_Source/faster-rcnn.py YOUR_CHECKPOINT_PATH --out ../work_dirs/faster-rcnn/LINZ2UGRC/faster-rcnn_train_real_linz_test_real_linz/prediction.pkl
``` 

Then, we label synthetic source domain data.  
We first need to predict pseudo labels for synthetic source domain data. To prevent error during testing, you can create an empty annotation file (does not contain bounding box annotations) for synthetic source domain data: 
```
python data_annotation/build_empty_annotation.py \
    --image-dir Data/Synthetic/LINZ-with-cars/images \
    --save-dir Data/Synthetic/LINZ-with-cars/annotations_coco_Empty.json \
    --coco-dir Data/Real/LINZ/test/annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
```
Then we provide an example command using faster-rcnn to label synthetic source domain data. You need to change the <b>data_root_test</b>, <b>test ann_file</b>, and <b>work_dir</b> variables to the ones corresponding to synthetic source domain data.  
```
python tools/test.py ../data_annotation/configs/Real_Source/faster-rcnn.py YOUR_CHECKPOINT_PATH --out ../work_dirs/faster-rcnn/LINZ2UGRC/faster-rcnn_train_real_linz_test_syn_linz/prediction.pkl
``` 
Then, please run the notebook [VisualizeTestResults.ipynb](VisualizeTestResults.ipynb) to determine the confidence threshold based on the maximum F1-score, and then run the notebook [ConvertPredToCOCOPseudoAnnotations.ipynb](ConvertPredToCOCOPseudoAnnotations.ipynb) label synthetic source domain data.   

Next, we train another detector on synthetic source domain cross-attention maps with pseudo labels. We provide an example command using faster-rcnn:  
```
cd mmdetection
# Train on synthetic source domain cross-attention maps
python tools/train.py ../data_annotation/configs/Synthetic_Heatmap/faster-rcnn.py
``` 

Then, we label synthetic target domain cross-attention maps. Similarly, you need to create an empty annotation file for synthetic target domain cross-attention maps by running [build_empty_annotation.py](build_empty_annotation.py). Next, we provide an example using faster-rcnn: 
```
python tools/test.py ../data_annotation/configs/Synthetic_Heatmap/faster-rcnn.py YOUR_CHECKPOINT_PATH --out ../work_dirs/faster-rcnn/LINZ2UGRC/faster-rcnn_train_syn_linz_hmap_test_syn_ugrc_hmap/prediction.pkl
```
To determine the confidence threshold, you can either run [VisualizeTestResults.ipynb](VisualizeTestResults.ipynb) and [ConvertPredToCOCOPseudoAnnotations.ipynb](ConvertPredToCOCOPseudoAnnotations.ipynb) similar to the previous step based on the training result on synthetic source domain cross attention maps (in this case you also need to obtain the prediction file for synthetic source domain cross attention maps), or run [refine_label.py](refine_label.py) that classifies objects with intermediate confidence score as described in the paper.
```
python data_annotation/refine_label.py \
--prediction_pkl work_dirs/faster-rcnn/LINZ2UGRC/faster-rcnn_train_syn_linz_hmap_test_syn_ugrc_hmap/prediction.pkl \
--synthetic_image_base_path Data/Synthetic/UGRC-with-cars/images \
--json_save_path Data/Synthetic/UGRC-with-cars/annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-FasterRCNN-SynUGRC-STACKDAAMHeatMaps-Clf-Refine.json \
--checkpoint_save_path work_dirs/faster-rcnn/LINZ2UGRC/heatmap-clf \
--pos_thresh 0.75 \
--neg_thresh 0.35 \
--hard_neg_thresh 0.05
```

Finally, we train the third detector on synthetic target domain images with pseudo labels, and then test on real target domain data. You also need to run [build_empty_annotation.py](build_empty_annotation.py) to create an annotation file for synthetic target domain background-only images.
```
cd mmdetection
# Train on synthetic target domain images
python tools/train.py ../data_annotation/configs/Synthetic_Target/faster-rcnn.py
``` 
Note: If you encounter ViT checkpoint loading issue when using ViTDet model, you might change the <b>_state_dict</b> variable in line 434 from ckpt['model'] to ckpt['state_dict'] in [vit.py](../mmdetection/projects/ViTDet/vitdet/vit.py).




