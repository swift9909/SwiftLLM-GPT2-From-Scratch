# SwiftLLM-GPT2-From-Scratch 🤖📝

## Overview
This repository contains a comprehensive PyTorch implementation of the GPT-2 language model, featuring:
- Full GPT-2 architecture implementation
- Automatic pre-trained weight downloading
- Custom model initialization
- Text generation capabilities
- Fine-tuning support

## 🚀 Features
- Complete GPT-2 model architecture
- Support for multiple model sizes (124M, 355M, 774M, 1558M)
- Pre-trained weight loading
- Flexible text generation
- Easy fine-tuning mechanism

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- PyTorch
- TensorFlow
- Tiktoken

### Installation
```bash
pip install torch tensorflow numpy requests tqdm tiktoken
```

## 🔧 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/gpt2-implementation.git
cd gpt2-implementation
```

### 2. Initialize Model
```python
model_size = "124M"  # Choose from "124M", "355M", "774M", "1558M"
models_dir = "./models"
model = initialize_gpt2_with_params(model_size, models_dir)
```

## 📝 Usage Examples

### Text Generation
```python
start_tokens = torch.tensor([[50256]])  # GPT-2 start token
generated_text = generate_text(model, start_tokens, max_length=50)
```

### Fine-tuning
```python
# Prepare your dataset
dataset = CustomDataset(texts, tokenizer, max_length=1024)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Fine-tune model
fine_tune(model, dataloader, epochs=3)
```

## 🛠 Customization
- Easily modify model hyperparameters
- Support for different model sizes
- Flexible architecture for various NLP tasks

## 📚 Model Sizes Supported
- 124M parameters
- 355M parameters
- 774M parameters
- 1558M parameters

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 🌟 Acknowledgements
- OpenAI for the GPT-2 model
- PyTorch Community
- [Vizuara's YouTube Playlist](https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu) for educational content on GPT-2 implementation

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/51308642/52ad60fe-0b66-4955-8390-cffcbecd8c0a/paste.txt
[2] https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu
