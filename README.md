# AGenDA
This is the official code for our ICCV 2025 paper:
> [Adapting Vehicle Detectors for Aerial Imagery to Unseen Domains with Weak Supervision](https://humansensinglab.github.io/AGenDA/)  
> Xiao Fang, Minhyek Jeon, Zheyang Qin, Stanislav Panev, Celso M de Melo, Shuowen Hu, Shayok Chakraborty, Fernando De la Torre

## Requirement
Under construction.
```
# Create virtual environment
conda create -n agenda python=3.9

# Install torch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

## Data preparation
Please follow the instruction [here](Data/README.md).

## Usage
### Stage 1: Data generation
Please follow the instruction [here](data_generation/README.md)

### Stage 2: Data annotation
Under construction.

## Citation
Please cite the paper if you use the code and datasets.
```
@misc{fang2025adaptingvehicledetectorsaerial,
      title={Adapting Vehicle Detectors for Aerial Imagery to Unseen Domains with Weak Supervision}, 
      author={Xiao Fang and Minhyek Jeon and Zheyang Qin and Stanislav Panev and Celso de Melo and Shuowen Hu and Shayok Chakraborty and Fernando De la Torre},
      year={2025},
      eprint={2507.20976},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.20976}, 
}
```

## Acknowledgement
The code is built on [diffusers](https://github.com/huggingface/diffusers/tree/main/examples), [DAAM](https://github.com/castorini/daam), and [AttnDreamBooth](https://github.com/lyuPang/AttnDreamBooth), thanks for their amazing work!
