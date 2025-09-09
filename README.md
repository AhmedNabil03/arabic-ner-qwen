# Fine-Tuning Qwen2.5-0.5B-Instruct on Arabic NER Dataset

## Overview
This project fine-tunes the **Qwen2.5-0.5B-Instruct** model on an Arabic Named Entity Recognition (NER) dataset using **Unsloth** and **QLoRA** to optimize efficiency. The workflow includes dataset preprocessing, fine-tuning, evaluation, and inference, ensuring an optimized and effective NLP pipeline.

---

## Features
- **Arabic NER Dataset**: Preprocessing and formatting the Arabic text data for NER tasks.
- **Pydantic Schema**: Structured entity extraction with strict schema adherence.
- **Unsloth Integration**: Leveraging Unsloth for high-speed fine-tuning.
- **QLoRA Optimization**: Efficient fine-tuning with memory optimization.
- **Model Deployment**: Prepared for inference and evaluation after fine-tuning.

---

## Model & Dataset
- The finetuned model is uploaded to the HuggingFaceHub: **Finetuned Model** [Arabic NER Qwen Model](https://huggingface.co/AhmedNabil1/arabic_ner_qwen_model)
- The processed data which is used for training: **Processed Dataset** [wojood-arabic-ner](https://huggingface.co/datasets/AhmedNabil1/wojood-arabic-ner)

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

## Fine-Tuning Qwen2.5-0.5B-Instruct
### 1. Tokenization and Formatting
- The dataset is tokenized using the **Qwen2.5-0.5B-Instruct** tokenizer.
- Tokenized data is prepared in a format suitable for model training.

### 2. Load Model and Fine-Tune with Unsloth
- The **Qwen2.5-0.5B-Instruct** model is loaded with **Unsloth**.
- Fine-tuning is performed using **QLoRA** for optimized memory efficiency.

---

## References & Base Resources
- The base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- The official dataset repository: **SinaLab/ArabicNER** [ArabicNER Dataset](https://github.com/SinaLab/ArabicNER)
  
---

## Repository Structure
```
arabic-ner-qwen/
├── qwen_finetuning.ipynb
└── README.md
```
