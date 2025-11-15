# Medical QLoRA Fine-Tuning with Unsloth

This project fine-tunes a medical language model using QLoRA and Unsloth in Google Colab. 

## Features
- Loads a prebuilt 4-bit LLaMA-3 model using Unsloth
- Uses a medical dataset (flashcards / clinical Q&A)
- Adds LoRA adapters for memory-efficient fine-tuning
- Trains the model with SFTTrainer
- Saves the fine-tuned adapter
- Generates medical responses to unseen queries

## How to Use
1. Open the Colab notebook `Medical_QLoRA_FineTuning_Unsloth.ipynb`
2. Install required packages
3. Run the notebook step by step to train and test the model

## Requirements
- Python 3.10+
- GPU with CUDA support
- Packages: `unsloth`, `transformers`, `datasets`, `trl`, `peft`, `torch`, `bitsandbytes`
