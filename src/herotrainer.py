import os
import numpy as np
import torch
import torch.nn as nn
import math
from utils import load_data, plot_losses, load_json
from heropicker import HeroPicker

torch.manual_seed(39)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = '/Home/siv33/vbo084/DotaPicker/data/processed/'
model_dir = '/Home/siv33/vbo084/DotaPicker/saved_models/'

fnames = ['dota_train_23.pt', 'dota_val_23.pt', 'dota_test_23.pt']
specnames = ['dota_matches.json']

dset_specs = load_json(os.path.join(data_dir, specnames[0]))
print(dset_specs)
VOCAB_LEN = dset_specs['simple_vocab_len'] + 15

config_dict = {'emb_dim': 1024,
               'dropout': 0.1,
               'vocab_len': VOCAB_LEN,
               'num_heads': 8,
               'num_dec_layers': 8,
               'dim_ff': 2048,
               'epochs': 5000,
               'lr': 0.0000001
               }

BATCH_SIZE = 1024
train_dloader, val_dloader, test_dloader = load_data(data_dir, fnames, batch_size=BATCH_SIZE)

def train_causal(traindloader, valdloader, config_dict, device):

    epochs = config_dict['epochs']
    model = HeroPicker(vocab_len=config_dict['vocab_len'], emb_dim=config_dict['emb_dim'],
                       num_heads=config_dict['num_heads'], dim_ff=config_dict['dim_ff'],
                       dropout_rate=config_dict['dropout'], num_decod_layers=config_dict['num_dec_layers'],
                       device=device
                       )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'])
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0 
    patience = 3  

    model = model.to(device)

    # Training and validation loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch in traindloader:
            inputs = batch
            targets = inputs[:, 1:]
            targets = targets.reshape(targets.size(0) * targets.size(1))
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs[:, :-1, :]
            outputs = outputs.reshape(outputs.size(0) * outputs.size(1), outputs.size(2))

            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(traindloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in valdloader:
                inputs = batch
                targets = inputs[:, 1: ]
                targets = targets.reshape(targets.size(0) * targets.size(1))
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                outputs = outputs[:, :-1, :]
                outputs = outputs.reshape(outputs.size(0) * outputs.size(1), outputs.size(2))
        
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valdloader)
        val_losses.append(avg_val_loss)

        # Check if validation loss has increased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # Reset the counter if validation loss improves
        else:
            early_stop_counter += 1  # Increment the counter if validation loss worsens
        
        # Early stopping condition
        if early_stop_counter >= patience:
            print(f"Validation loss increased for {patience} consecutive epochs. Stopping training.")
            torch.save(model, os.path.join(model_dir, f'heropicker_earlystop_{epoch+1}epcs.pt'))
            break

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print('Saving final model.')
    torch.save(model, os.path.join(model_dir, f'heropicker_final_23.pt'))

    return model, train_losses, val_losses

print(f'Starting to train HeroPicker.')
model, tls, vls = train_causal(val_dloader, test_dloader, config_dict, device)
plot_losses(tls, vls)