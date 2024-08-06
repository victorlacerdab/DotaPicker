import json
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

def match_picks_cleaner(dframe: DataFrame) -> DataFrame:
    dframe = dframe[['is_pick', 'hero_id', 'team', 'order', 'match_id']]
    return dframe

def match_results_cleaner(dframe: DataFrame) -> DataFrame:
    dframe = dframe[['match_id', 'radiant_win']]
    return dframe

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return dict(data)

class DotaTokenizer:

    '''
    self.simple_vocab (Dict): a vocabulary containing the special tokens provided by the user and int: name pairs.
    self.pickban_vocab (Dict): a vocabulary containing the special tokens provided by the user, but which has
                               double the amount of tokens per hero, one for when the hero was picked, and one for
                               when the hero was banned (e.g. n: [PICK_techies], n+1: [BAN_techies]).
    '''

    def __init__(self, metadata: dict, special_tks: list):
        self.simple_vocab = self.get_vocab(metadata, special_tks, alt_vocab=False)
        self.pickban_vocab = self.get_vocab(metadata, special_tks, alt_vocab=True)
        self.simple_ttoi = {v: k for k, v in self.simple_vocab.items()}
        self.alt_ttoi = {v: k for k, v in self.pickban_vocab.items()}
    
    def encode(self, match: str, alt: bool) -> list:
        if not alt:
            match = [self.simple_ttoi[token] for token in match]
        else:
            match = [self.alt_ttoi[token] for token in match]
        
        return match

    def decode(self, match: list, alt: bool) -> list:
        if not alt:
            match = [self.simple_vocab[tok_idx] for tok_idx in match]
        else:
            match = [self.pickban_vocab[tok_idx] for tok_idx in match]
        
        return match

    def get_vocab(self, metadata: dict, special_tks: list, alt_vocab: bool) -> dict:
        
        vocab = {}
        idx_to_ignore = len('npc_dota_hero_')
        counter = 0

        for k,v in metadata.items():
            v = v['name'][idx_to_ignore:]
            if not alt_vocab:
                vocab.update({int(k): v})
            else:
                vocab.update({counter: f'[PICK_{v}]'})
                counter += 1
                vocab.update({counter: f'[BAN_{v}]'})
                counter += 1

        if not alt_vocab:
            for i, tk in enumerate(special_tks):
                vocab.update({i+len(metadata.items()): tk})
        else:
            for i, tk in enumerate(special_tks):
                vocab.update({i+counter: tk})
                counter += 1

        return vocab


class Dotaset(Dataset):
    def __init__(self, samples):
        self.samples = torch.tensor(samples, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]