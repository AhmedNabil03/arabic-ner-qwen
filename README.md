# Fine-Tuning Qwen2.5-1.5B-Instruct on Arabic NER Dataset

## Overview
This project fine-tunes the **Qwen2.5-1.5B-Instruct** model on an Arabic Named Entity Recognition (NER) dataset using **Unsloth** and **QLoRA** to optimize efficiency. The workflow includes dataset preprocessing, fine-tuning, evaluation, and inference, ensuring an optimized and effective NLP pipeline.

## Features
- **Arabic NER Dataset**: Preprocessing and formatting of Arabic text data for NER tasks.
- **QLoRA Optimization**: Efficient fine-tuning with memory optimization.
- **Unsloth Integration**: Leveraging Unsloth for high-speed fine-tuning.
- **Pydantic Schema**: Structured entity extraction with strict schema adherence.
- **Model Deployment**: Prepared for inference and evaluation after fine-tuning.

---

## Model & Dataset
- The finetuned model is uploaded to the HuggingFaceHub: **Finetuned Model**: [Arabic NER Qwen Model](https://huggingface.co/AhmedNabil1/arabic_ner_qwen_model)
- The official dataset repository: **SinaLab/ArabicNER** [ArabicNER Dataset](https://github.com/SinaLab/ArabicNER)

---

## Installation & Setup
### 1. Install Dependencies
Ensure you have the required dependencies installed:
```bash
!apt-get install git
!pip install -qU datasets transformers protobuf unsloth
```

### 2. Clone the Dataset Repository
```bash
!git clone https://github.com/SinaLab/ArabicNER.git
```

### 3. Mount Google Drive (If using Colab)
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## Data Preprocessing
### 1. Load and Structure NER Data
- The dataset is loaded from text files, where each sentence is tokenized, and labels are extracted.
- Each token is assigned a corresponding NER label.

### 2. Convert to DataFrame Format
- The structured dataset is transformed into a DataFrame.
- Each row consists of a text sentence and its corresponding NER-labeled entities.

### 3. Save Processed Data
- The preprocessed dataset is saved as CSV files for easy access and training.

### 4. Convert Data for Training
- The processed dataset is converted into a **Hugging Face DatasetDict** format.
- Training and validation datasets are structured appropriately.

### 5. Save and Load Dataset
- The datasets are stored on Google Drive and can be reloaded for training.
---

## Fine-Tuning Qwen2.5-1.5B-Instruct
### 1. Tokenization and Formatting
- The dataset is tokenized using the **Qwen2.5-1.5B-Instruct** tokenizer.
- Tokenized data is prepared in a format suitable for model training.

### 2. Load Model and Fine-Tune with Unsloth
- The **Qwen2.5-1.5B-Instruct** model is loaded with **Unsloth**.
- Fine-tuning is performed using **QLoRA** for optimized memory efficiency.

---

## Repository Structure
```
arabic-NER-Qwen-finetuning/
├── data/
│   ├── train.txt 
│   ├── test.txt
│   └── val.txt
│
├── csv_files/
│   ├── train.csv
│   ├── test.csv
│   └── val.csv
│
├── datasets/
│   ├── train_dataset
│   └── val_dataset
│
├── qwen_finetuning.ipynb
└── README.md
```
