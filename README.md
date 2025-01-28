# MMISA-KM
## Table of Contents

1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [script](#script)
---
## 1. Introduction

We proposed MMISA-KM, a novel deep learning model designed to predict the Michaelis constant ($K_{m}$) using protein sequences, protein graphs, substrate SMILES strings, and molecular graphs. MMISA-KM consists of feature extraction, feature fusion, and output modules. The feature extraction module employs convolutional neural networks (CNNs) and graph neural networks (GNNs) to extract both sequence-based and graph-based features from proteins and substrates. The feature fusion module integrates these features using self-attention mechanisms, while the output module utilizes a multi-layer perceptron (MLP) to generate $K_{m}$ predictions. MMISA-KM was trained and evaluated on BRENDA and SABIO-RK dataset, demonstrating superior performance compared to existing state-of-the-art models and variant models.


## 2. Python Environment

Python 3.9 and packages version:

- pytorch==2.2.1
- tqdm==4.66.2                            
- torchvision==0.17.1    
- transformers==4.22.2
- numpy==1.26.4
- pandas==2.1.4
- scikit-learn==1.4.1
- scipy==1.12.0 

## 3. Project Structure

### 3.1 **Dataset**

  The BRENDA and SABIO-RK databases are two prominent resources for enzyme-related information. We use the dataset collected by MPEK from the BRENDA and SABIO-RK databases as the baseline dataset. To ensure data quality, several rounds of data cleaning were performed and the final dataset contained 24,585 unique $Km$ entries, and then the dataset was divided into training, validation, and testing datasets according to a ratio of 8:1:1. As a result, there were 19668 training samples, 2459 validation samples, and 2458 testing samples. Each sample comprises a protein sequence, a substrate SMILES string, and a $Km$ value. We set the lengths of protein sequences and SMILES strings to be 500 and 100, respectively. Sequences with a longer length were truncated, and sequences with a shorter length were padded with zeros to reach the fixed length. 

### 3.2 **Model**
   -  The overall architectures of MMISA-KM is presented in the following figure, which consists of a feature extraction module, a feature fusion module and an output module.
   -  ![Model Architecture](https://github.com/aoteman250/MMISA-KM/blob/main/MMISA-KM.jpg)
   -  The ESM-2 model is available at (https://github.com/facebookresearch/esm).
   -  To load the model from Huggingface, we can use the following code:
```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
inputs = tokenizer(smiles, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
outputs.pooler_output
```

### 3.3 **dataset**
   -  The .csv files contain the datasets utilized in our study.
   - `contactmap.py` converts protein sequences into protein contactmap.
### 3.4 **script**
   -   To train the model, we can run `main.py` script using the train and valid dataset.
   -   We can also run `test.py` to test the model.
   - `data_process.py` is the data preparation phase.
   - `utils1.py` is the model preparation phase.
   -  trained_model.pt is the MMISA-KM model that is trained on the training subest of the BRENDA and SABIO-RK dataset.
   - `model.py` implements the MMISA-KM which consists of a feature extraction module, a feature fusion module and an output module.
