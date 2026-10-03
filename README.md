# PPSB-NeurIPS-26
This is code for paper "Progressive Pseudo-label Self-balancing Towards Unsupervised Vision-Language Models Adaptation" in NeurIPS 2026.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Reproducing the Main Results](#reproducing-the-main-results)
  - [Data Preparation](#data-preparation)
  - [Running the Experiments](#running-the-experiments)

## Environment Setup

In this project, we use Python 3.9.12 and the dependencies listed in the `requirements.txt` file. To install all the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Additionally, you need to manually install the [dassl](https://github.com/KaiyangZhou/Dassl.pytorch) library in your environment by running the by following command:

```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
python setup.py develop
```

## Reproducing the Main Results

### Data Preparation

To reproduce the results in Table 1 of our paper, you can download the following four datasets (Flowers102, FGVC-Aircraft, CUB, and DTD) from [HERE](https://www.kaggle.com/datasets/konszer/cpl-datasets), OxfordPets from [HERE](https://www.robots.ox.ac.uk/~vgg/data/pets/), and StanfordCars from [HERE](https://www.modelscope.cn/datasets/tanganke/stanford_cars/files). We use the train and test splits of the dataset suggested splits. 

Then, before running the experiments, create the following folders to save prompts, datasets, and results:

```bash
mkdir output
mkdir data_
mkdir Features
mkdir script_results
mkdir trained_prompts
```

After downloading and moving the datasets to the `data_` folder, the folder structure should be as follows:

```
data_
├── CUB
│   └── CUB_200_2011
│       ├── ...
│       └── ...
├── DTD
│   ├── ...
│   └── ...
├── FGVCAircraft
│   ├── ...
│   └── ...
├── Flowers102
│   ├── ...
│   └── ...
├── OxfordPets
│   ├── ...
│   └── ...
├── StanfordCars
│   ├── ...
│   └── ...
```

### Running the Experiments

To execute the training strategies employing PPSB across text prompts and visual prompts, run the following commands:

- For text prompt tuning (including UL, SSL, and TRZSL settings):  
  
  ```bash
  CUDA_VISIBLE_DEVICES=[...] bash scripts/run-textPT_ALL.sh "0.01"
  ```

- For visual prompt tuning (including UL, SSL, and TRZSL settings):  
  
  ```bash
  CUDA_VISIBLE_DEVICES=[...] bash scripts/run-visualPT_ALL.sh "0.01"
  ```

Replace `[...]` with the GPU number you want to use, and "0.01" represents the value of the default learning rate. Here is an example command:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run-textPT_ALL.sh "0.01"
```

Some notes:

- The running logs from the above scripts are automatically saved in the `output/` directory. 
- The results will be collected by dataset name and saved in `script_results/`. 
- In `trained_prompts/`, the prompt parameters at each iteration and the final prompts after training will be saved. 
- In `Features/`, the features extracted from the images by the vision encoder for each dataset will be pre-saved. This can accelerate the training process of text prompt tuning and is enabled by default. 

For more information about the hyperparameters and settings, you can refer to the comments in the `run-textPT_ALL.sh` and `run-visualPT_ALL.sh` scripts.

[Back to Table of Contents](#table-of-contents)
