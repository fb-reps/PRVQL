# PRVQL

[![Watch Demo](https://img.shields.io/badge/Watch-Demo-blue?style=for-the-badge)](https://youtu.be/_80-whm9e0I)

## 📝 Overview

The model and code will be public soon.

### PRVQL Core Idea
Below is an overview of the core idea behind PRVQL:

![Compare PRVQL with current methods](docs/fig1.png)

### PRVQL Model Framework
Below is the model framework for PRVQL:

![Framework](docs/fig2.png)

---

## 🚀 Environment Setup

To set up the environment, follow these steps:

```bash
conda create --name prvql python=3.8 -y
conda activate prvql

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install -r requirements.txt
```

---

## 📦 Pretrained Weights

Please download the pretrained weight from [Google Drive](https://drive.google.com/file/d/1A6JTGqG32Ja-4Qo2T44KIcL74hcKEo8T/view?usp=drive_link) and place it in:
```bash
./output/ego4d_vq2d/train/train
```

---

## 📂 Dataset Preparation

### 1️⃣ Process the Dataset
Follow the instructions in the [VQLoC Repository](https://github.com/hwjiang1510/VQLoC/blob/main/README.md#download-dataset) to process the dataset into video clips and images.

### 2️⃣ Restructure the Dataset
Ensure your dataset is structured as follows:

```plaintext
./your/dataset/path/
└── datav2
    ├── clips
    │   ├── 1.mp4
    │   └── ...
    ├── images
    │   ├── 1   
    │   │   ├── 1.mp4
    │   │   └── ...
    │   └── ...        
    ├── train_annot.json
    ├── val_annot.json
    ├── vq_test_unannotated.json
    ├── vq_train.json
    └── vq_val.json
```

### 3️⃣ Update Configuration Files
Modify the dataset path in the following configuration files:

- `config/eval.yaml`
- `config/train.yaml`
- `config/val.yaml`

Update the root path:

```yaml
root: './your/dataset/path/'
```

---

## 🏋️‍♂️ Training & Evaluation

### 🔥 Train the Model
```bash
sh ./train.sh
```

### ✅ Validate the Model
```bash
sh ./test.sh
```

### 🧪 Test and Submit Results
```bash
sh ./submit_evalai.sh
```
After running this command, you will get the latest tracks in a `.json` file. Submit it to the [EvalAI Challenge](https://eval.ai/web/challenges/challenge-page/1843/submission) to get the final result.

---

## 📊 Benchmark: Ego4D Validation and Test Sets

### Validation Set

| **Methods**      | **tAP$_{25}$** | **stAP$_{25}$** | **rec%** | **Succ** |
|-----------------|--------------|---------------|--------|--------|
| STARK *(ICCV'21)* | 0.10 | 0.04 | 12.41 | 18.70 |
| SiamRCNN *(CVPR'22)* | 0.22 | 0.15 | 32.92 | 43.24 |
| NFM *(VQ2D Challenge'22)* | 0.26 | 0.19 | 37.88 | 47.90 |
| CocoFormer *(CVPR'23)* | 0.26 | 0.19 | 37.67 | 47.68 |
| VQLoC *(NeurIPS'23)* | 0.31 | 0.22 | 47.05 | 55.89 |
| **PRVQL (Ours)** | **0.35** | **0.27** | **47.87** | **57.93** |

### Test Set

| **Methods**      | **tAP$_{25}$** | **stAP$_{25}$** | **rec%** | **Succ** |
|-----------------|--------------|---------------|--------|--------|
| STARK *(ICCV'21)* | - | - | - | - |
| SiamRCNN *(CVPR'22)* | 0.20 | 0.13 | - | - |
| NFM *(VQ2D Challenge'22)* | 0.24 | 0.17 | - | - |
| CocoFormer *(CVPR'23)* | 0.25 | 0.18 | - | - |
| VQLoC *(NeurIPS'23)* | 0.32 | 0.24 | 45.11 | 55.88 |
| **PRVQL (Ours)** | **0.37** | **0.28** | **45.70** | **59.43** |

📌 For any issues, please open an [issue](https://github.com/fb-reps/PRVQL/issues) or reach out to the maintainers.

