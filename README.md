# SwiftLLM-GPT2-From-Scratch

This repository contains a PyTorch implementation of the GPT-2 architecture from scratch. It includes functionality for downloading pre-trained weights, fine-tuning on custom datasets, and generating text. The implementation is modular and allows for easy modification of hyperparameters or model architecture.
Features
Full implementation of the GPT-2 architecture in PyTorch.
Automatic downloading of pre-trained weights from OpenAI's public storage.
Fine-tuning support for custom datasets.
Text generation using the pre-trained or fine-tuned model.
Requirements
Before running the code, ensure you have the following installed:
Python Version
Python 3.8 or higher
Required Libraries
Install the required libraries using pip:
bash
pip install torch tensorflow numpy requests tqdm tiktoken
How to Use
1. Clone the Repository
Clone this repository to your local machine:
bash
git clone https://github.com/your-username/gpt2-from-scratch.git
cd gpt2-from-scratch
2. Download Pre-trained Weights
The script automatically downloads pre-trained weights for GPT-2 when you run it. You can specify the model size (124M, 355M, 774M, or 1558M) in the code.
Example:
python
model_size = "124M"  # Choose from "124M", "355M", "774M", "1558M"
models_dir = "./models"  # Directory to store downloaded models
model = initialize_gpt2_with_params(model_size, models_dir)
3. Fine-tune on Custom Data
To fine-tune the model on your dataset:
Prepare Your Dataset
Create a list of text samples (e.g., sentences or paragraphs).
Use tiktoken to tokenize your data and feed it into a PyTorch DataLoader.
Example:
python
from torch.utils.data import DataLoader

texts = ["This is an example sentence.", "Another example for training."]
tokenizer = tiktoken.get_encoding("gpt2")
max_length = 1024

# Create dataset and dataloader
dataset = GPT2Dataset(texts, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
Fine-tune the Model
Use the provided fine_tune function to train the model on your dataset:
python
fine_tune(model, dataloader, epochs=3)
Fine-tuning Code Example:
python
def fine_tune(model, data_loader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
4. Generate Text
Use the generate_text function to generate text from a starting token.
Example:
python
start_tokens = torch.tensor([[50256]])  # Start token for GPT-2 (e.g.,
