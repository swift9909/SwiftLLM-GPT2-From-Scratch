import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Hyperparameters and Configuration
vocab_size = 50257
d_model = 768
n_heads = 12
n_layers = 12
max_seq_len = 1024

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, d_model * 4) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.token_embedding(x) + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## We have reached here until now ---> we have downloaded the files on our local machine.

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)       # load the model check point
    settings = json.load(open(os.path.join(model_dir, "hparams.json"))) #load the confg settings
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings) #load the paramters using checkpoint and settings

    return settings, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def initialize_gpt2_with_params(model_size, models_dir):
    settings, params = download_and_load_gpt2(model_size, models_dir)
    
    global d_model, n_heads, n_layers
    d_model = settings['n_embd']
    n_heads = settings['n_head']
    n_layers = settings['n_layer']
    
    model = GPT2(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    
    # Load parameters into the model
    for name, param in model.named_parameters():
        if name.startswith('token_embedding'):
            param.data = torch.from_numpy(params['wte'])
        elif name.startswith('positional_encoding'):
            param.data = torch.from_numpy(params['wpe'])
        elif name.startswith('layers'):
            layer_num = int(name.split('.')[1])
            if 'self_attn.W_q.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['attn']['c_attn']['w'][:,:d_model]).t()
            elif 'self_attn.W_k.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['attn']['c_attn']['w'][:,d_model:2*d_model]).t()
            elif 'self_attn.W_v.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['attn']['c_attn']['w'][:,2*d_model:]).t()
            elif 'self_attn.W_o.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['attn']['c_proj']['w']).t()
            elif 'feed_forward.linear1.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['mlp']['c_fc']['w']).t()
            elif 'feed_forward.linear2.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['mlp']['c_proj']['w']).t()
            elif 'norm1.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['ln_1']['g'])
            elif 'norm2.weight' in name:
                param.data = torch.from_numpy(params['blocks'][layer_num]['ln_2']['g'])
        elif name.startswith('fc_out'):
            param.data = torch.from_numpy(params['wte'].T)
    
    return model

# Usage
model_size = "124M"
models_dir = "./models"
model = initialize_gpt2_with_params(model_size, models_dir)

# Training example
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Generation example
def generate_text(model, start_tokens, max_length=50):
    model.eval()
    current_tokens = start_tokens
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(current_tokens)
            next_token = torch.argmax(outputs[:, -1, :])
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
    return current_tokens


