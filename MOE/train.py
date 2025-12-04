import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset

# Assuming you have a separate file with model architecture
from model import TritonTransformerLM, TritonAttention


class ModelConfig:
    vocab_size = 50257  # GPT-2 default
    hidden_size = 512
    num_attention_heads = 8
    num_hidden_layers = 8
    max_position_embeddings = 1024

import os
from datasets import load_dataset, DatasetDict

def prepare_data(tokenizer, block_size=128, subset_size=10000, cache_dir='cached_tokenized'):
    # If a cached tokenized dataset exists, load it.
    if os.path.exists(cache_dir):
        print("Loading tokenized dataset from disk...")
        tokenized_datasets = DatasetDict.load_from_disk(cache_dir)
    else:
        print("Tokenizing dataset...")
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                truncation=True,
                max_length=block_size,
                padding='max_length'
            )
    
        # Load WikiText dataset
        dataset = load_dataset('wikitext', 'wikitext-103-v1')
        
        # Optionally, reduce the dataset size for quick experiments.
        # This selects the first `subset_size` examples from the training and validation splits.
        dataset['train'] = dataset['train'].select(range(min(subset_size, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(subset_size, len(dataset['validation']))))
        
        # Tokenize the dataset
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        # Save the tokenized dataset to disk for later reuse
        tokenized_datasets.save_to_disk(cache_dir)
        print("Tokenized dataset saved to disk.")

    # Prepare DataLoaders
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    
    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        shuffle=True,
        batch_size=8,
        collate_fn=default_data_collator
    )
    
    val_dataloader = DataLoader(
        tokenized_datasets['validation'],
        shuffle=False,
        batch_size=8,
        collate_fn=default_data_collator
    )
    
    return train_dataloader, val_dataloader


from tqdm import tqdm

def train(model, train_dataloader, val_dataloader, device):
    # Optimization setup
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(5):  # 5 epochs
        total_train_loss = 0
        
        # Training phase with tqdm progress bar
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch in train_bar:
            # Prepare batch
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        # Print epoch summary for training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/5")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase with tqdm progress bar
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                labels = input_ids.clone()
                
                logits = model(input_ids)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                total_val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}\n")
        
        # Switch back to training mode
        model.train()
    
    # Save model
    torch.save(model.state_dict(), 'triton_transformer_lm.pt')
    print("Model training completed and saved.")

def main():
    import torch
    from transformers import AutoTokenizer
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data (with caching and subset reduction)
    train_dataloader, val_dataloader = prepare_data(tokenizer, subset_size=10000, cache_dir='cached_tokenized')
    
    # Model config and initialization
    config = ModelConfig()
    model = TritonTransformerLM(config).to(device)
    
    # Training
    train(model, train_dataloader, val_dataloader, device)

if __name__ == '__main__':
    main()