# LRRN: Lightweight Remote Sensing Image Recognition Network

This repository contains the official PyTorch implementation of the paper **"LRRN: Efficient Multi-Scale Feature Fusion for Lightweight Remote Sensing Image Recognition"** by Chen et al. (2025).

The code includes the proposed LRRN model (composed of FIN and CNH modules), baseline networks, training/testing scripts, and the HBUA-NR5 dataset used in transfer learning experiments.

**Paper link:** [https://github.com/ysuc/LRRN](https://github.com/ysuc/LRRN) (publicly available as stated in the paper)

## 📌 Overview

LRRN is a lightweight convolutional neural network designed for efficient remote sensing image recognition. It consists of two main modules:

*   **Feature Integration Network (FIN):** Extracts multi‑scale features using parallel convolutional paths and fuses them through skip connections (implemented in `modul_zoo.py` as class `FIN`).
*   **Classification Network Head (CNH):** Aggregates sparse discriminative features via a Tandem Pooling mechanism and performs final classification (implemented in `modul_zoo.py` as class `CNH`).

The model achieves competitive accuracy on multiple public datasets while having an extremely low parameter count (only **2.07M**) and fast inference speed, making it suitable for deployment on resource‑constrained devices.

## 📁 Repository Structure

```text
LRRN/
├── README.md
├── Main.py                         # End‑to‑end training of FIN + CNH (LRRN full model)
├── Baseline_network_testing.py     # Train/test baseline models (ResNet, MobileNet, ViT, etc.)
├── Transfer_learning.py             # Fine‑tune pre‑trained models on HBUA‑NR5
├── Obtain_confusion_matrix.py       # Generate confusion matrices from trained models
├── Obtain_paramsize.py              # Compute parameter counts and GPU memory footprint
├── Feature_map_visualization.py     # Visualize feature maps from the FIN module
├── Build_dataset.py                  # Split dataset into train/val folders
├── ViT.py                            # Vision Transformer implementation (for comparison)
├── modul_zoo.py                      # Contains all model definitions (FIN, CNH, baselines, etc.)
├── LRRN_modul.py                      # Alternative implementation of LRRN as a single class
├── PTH/                               # Pre‑trained weights
│   ├── NWPU-RESISC45/                 # Trained on NWPU‑RESISC45
│   │   ├── FIN.pth
│   │   └── CNH.pth
│   ├── EuroSAT/                       # Trained on EuroSAT
│   ├── RSSCN7/                         # Trained on RSSCN7
│   └── HBUA-NR5/                       # Fine‑tuned on HBUA‑NR5
├── confusion_matrices/                 # Output confusion matrices (PDF/PNG files)
├── data/                               # Dataset directory (to be created by user)
│   └── HBUA-NR5.rar                    # Original HBUA‑NR5 dataset (extract before use)
└── requirements.txt                     # Python dependencies
```
## 📦 Datasets

### HBUA‑NR5 Dataset
The **HBUA‑NR5** dataset is a small‑scale self‑compiled benchmark used in the paper. It contains five scene categories: **coast**, **forest**, **desert**, **cloud**, **city**.  
The original 149 images were augmented to 298 via horizontal flipping. The dataset is provided as `data/HBUA-NR5.rar`. Please extract it before training:
```bash
# Install unrar if needed (Linux)
sudo apt-get install unrar

# Extract
unrar x data/HBUA-NR5.rar data/```bash
# Install unrar if needed (Linux)
sudo apt-get install unrar

# Extract
unrar x data/HBUA-NR5.rar data/
```
After extraction, the folder should be structured as follows (used by `Build_dataset.py`):
```text
data/HBUA-NR5/
├── coast/
├── forest/
├── desert/
├── cloud/
└── city/
```
### Other Public Datasets
The paper also evaluates LRRN on:

NWPU‑RESISC45 (31,500 images, 45 classes)

EuroSAT (~27,000 images, 10 classes)

RSSCN7 (2,800 images, 7 classes)

You can download these datasets from their official sources and organize them into train/val subfolders (e.g., using `Build_dataset.py`). The expected structure for `ImageFolder` is:
```text
data/<dataset_name>/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```
## 🛠️ Installation
### Prerequisites
Python 3.8+

PyTorch 1.10+

CUDA (optional, for GPU acceleration)

### Steps
1.Clone the repository:
```bash
git clone https://github.com/ysuc/LRRN.git
cd LRRN
```
2.Create a virtual environment (recommended):
```bash
python -m venv lrrn_env
source lrrn_env/bin/activate   # Linux/Mac
lrrn_env\Scripts\activate      # Windows
```
Install dependencies:
```bash
pip install -r requirements.txt
```
### Dependencies
The requirements.txt includes:

torch

torchvision

numpy

matplotlib

scikit-learn

seaborn

einops (for ViT)

Pillow

tqdm

rarfile (for extracting RAR archives)

torchinfo (for parameter summary)

## 🚀 Usage
### 1. Configure the Experiment
All hyperparameters and paths are set in the `DefaultConfigs` class inside each script (e.g., `Main.py`, `Baseline_network_testing.py`). You must edit the placeholders (like "**Dataset save path**") to your actual paths before running.

Example from `Main.py`:
```python
class DefaultConfigs(object):
    data_dir = "path/to/your/dataset"          # e.g., "./data/NWPU-RESISC45"
    lr = 0.1
    epochs = 50
    num_classes = 10                            # change according to your dataset
    batch_size = 16
    # ... other settings
```
### 2. Train LRRN from Scratch (FIN + CNH)
Run Main.py to train the FIN and CNH modules jointly:

```bash
python Main.py
```
The script will:

Load training and validation data using `ImageFolder`

Train both modules for the specified number of epochs

Print training/validation loss and accuracy

Plot loss and accuracy curves at the end

Save the trained weights as `FIN.pth` and `CNH.pth` (you need to set the save paths in the code).

### 3. Test a Trained Model
To evaluate a trained LRRN model on the validation set, you can modify Main.py to load the saved weights and run only the validation loop, or write a simple testing script. A minimal example:

```python
import torch as t
import modul_zoo as mz

# Load modules
fin = mz.FIN().cuda()
cnh = mz.CNH().cuda()
fin.load_state_dict(t.load('PTH/NWPU-RESISC45/FIN.pth'))
cnh.load_state_dict(t.load('PTH/NWPU-RESISC45/CNH.pth'))

# ... dataloader ...
with t.no_grad():
    for x, y in dataloader['val']:
        x = x.cuda()
        f = fin(x)
        out = cnh(f)
        # compute accuracy
```
### 4. Train/Test Baseline Networks
Use `Baseline_network_testing.py` to train or test baseline models (ResNet, MobileNet, ViT, etc.). By default, it trains the CNH model (which is a baseline itself), but you can replace the model call with any model from `modul_zoo.py` or `ViT.py`.

```bash
python Baseline_network_testing.py
```
To test a pre‑trained baseline, modify the script to load weights and run only validation.

Supported models in `modul_zoo.py` include:

`resnet_18(num_classes)`

`resnet_50(num_classes)`

`vgg_13(num_classes)`

`ICN_l`, `eCNN_l`, `ICN_nocat`, `FC`, etc.

### 5. Transfer Learning on HBUA‑NR5
`Transfer_learning.py` demonstrates how to load a pre‑trained model (e.g., ConvNeXt) and fine‑tune only the final classifier on HBUA‑NR5.

```bash
python Transfer_learning.py
```
Make sure to set the correct paths for the pre‑trained weights and the dataset. The script currently uses `convnext_base`; you can replace it with any model from `modul_zoo.py` or `tv.models`.

### 6. Generate Confusion Matrix
After testing, produce a confusion matrix with Obtain_confusion_matrix.py:

```bash
python Obtain_confusion_matrix.py
```
You must set:

`test_dataset` path to your validation folder.

Paths to the trained `FIN.pth` and `CNH.pth` (or a single model for baselines).

Output file names (PDF/PNG).

The script generates both a normalized and an absolute count confusion matrix with optimized visualization for many classes (e.g., 45 classes in NWPU‑RESISC45).

### 7. Compute Model Parameters and Memory Footprint
`Obtain_paramsize.py` uses `torchinfo.summary` to print detailed layer information, total parameters, and memory usage.

```bash
python Obtain_paramsize.py
```
Modify the model creation line (e.g., net = mz.resnet_50(45)) to the model you want to analyze.

### 8. Visualize Feature Maps from FIN
Feature_map_visualization.py loads a trained FIN model and visualizes its output feature maps for images in a given folder.

```bash
python Feature_map_visualization.py
```
Set `data_from` to the folder containing test images, and `model.load_state_dict(torch.load('**FIN parameter save path**'))` to your trained FIN weights.

### 9. Prepare Dataset (Train/Val Split)
If your raw dataset is not already split, use `Build_dataset.py`:

```bash
python Build_dataset.py
```
Set `old_data` to the path containing class‑named subfolders, and `data` to the output directory. The script will randomly split 80% of images into `train` and 20% into `val`.

## 📊 Pre‑trained Weights
The PTH/ folder contains weights for LRRN trained on three public datasets and fine‑tuned on HBUA‑NR5.
Each subfolder includes two files:

`FIN.pth` : weights of the Feature Integration Network

`CNH.pth` : weights of the Classification Network Head

To use them for inference, load both modules as shown in the testing example above.
## 📈 Experimental Results (from the paper)
| Dataset | Accuracy (%) | Parameters (M) | Inference Time (ms) |
| :--- | :---: | :---: | :---: |
| NWPU-RESISC45 | 94.81 | 2.06 | 49.55 |
| EuroSAT | 96.44 | 2.06 | 49.55 |
| RSSCN7 | 83.75 | 2.06 | 49.55 |
| HBUA-NR5 (TL) | 86.96 | 2.06 | 49.55 |

Comparison with baselines: LRRN achieves the highest Accuracy per Parameter (APP) on all datasets, demonstrating superior parameter efficiency. Detailed comparisons and confusion matrices are available in the paper and in the `confusion_matrices/` folder.
## 📝 Citation
If you find this work useful for your research, please cite our paper:

```bibtex
@article{chen2025lrrn,
  title={Efficient Multi-Scale Feature Fusion for Lightweight Remote Sensing Image Recognition},
  author={Chen, et al.},
  journal={GPS Solutions},
  year={2025}
}
```
## 🙌 Acknowledgements
We thank the providers of the NWPU‑RESISC45, EuroSAT, and RSSCN7 datasets for making their data publicly available.

For any questions or issues, please open an issue on GitHub or contact the authors.
