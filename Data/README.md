# Datasets
This page introduces how to download and use our released datasets.

## Introduction
We introduce two new real-world aerial view datasets, <b>LINZ</b> and <b>UGRC</b>, 
captured in [Selwyn (New Zealand)](https://data.linz.govt.nz/layer/51926-selwyn-0125m-urban-aerial-photos-2012-2013/) and [Utah (USA)](https://gis.utah.gov/products/sgid/aerial-photography/high-resolution-orthophotography/), respectively. Both datasets have ground sampling distance (GSD) of 12.5 cm per px and have been
sampled to *112 px × 112 px* image size. For data annotation, we label only the small vehicle centers. To leverage the abundance of bounding box-based open-source object detection frameworks, we define a fixed-size ground truth bounding box of *42.36 px × 42.36 px* center at each vehicle. Annotations are provided in COCO format *[x, y, w, h]*, where *"small"* in the annotation json files denotes the small vehicle class and *(x, y)* denotes the top-left corner of the bounding box.
We use *AP50* as evaluation metrics. For more details, please check our paper.

## Download
You can download LINZ and UGRC using this command:
```
wget https://datastore.shannon.humansensing.cs.cmu.edu/api/public/dl/IbKPrtJW -O aerial_dataset.zip
```
After unzipping the data, please organize it in the following format:
```
|-- Data
    |-- Real
        |-- LINZ
            |-- test
                |-- images
                    |-- 0001_0001_0000001.jpg
                    |-- ...
                |-- annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
            |-- train
                |-- images
                    |-- 0002_0001_0000001.jpg
                    |-- ...
                |-- annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
            |-- validation
                |-- images
                |-- annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
        
        |-- UGRC
            |-- test
                |-- images
                    |-- 12SVK260260_0000001.jpg
                    |-- ...
                |-- annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
            |-- train
                |-- images
                    |-- 12TVK100560_0000001.jpg
                    |-- ...
                |-- annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
            |-- validation
                |-- images
                |-- annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json
        
```

## Inference
We provide several examples of using open-set models to directly evaluate performance on our datasets.  
[inference](inference): Examples using Gemini, InternVL3, and DeepSeek-VL2 to inference on UGRC test set.  
[ConvertPseudoAnn.py](utils/ConvertPseudoAnn.py): Convert predicted bounding boxes to pseudo annotations.  
[EvaluatePseudoAnn.py](utils/EvaluatePseudoAnn.py): Compute precision and recall between ground truth and predicted results.
