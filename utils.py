import json

def data_cleaner(dframe):

    pass

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return dict(data)

class DotaTokenizer:
    def __init__(self, metadata: dict, special_tks: list):
        self.vocab = self.get_vocab(metadata, special_tks)
    
    def encode(self):
        pass

    def decode(self):
        pass

    def get_vocab(self, metadata: dict, special_tks: list) -> dict:
        
        vocab = {}

        for i, tk in enumerate(special_tks):
            vocab.update({i: tk})

        for k,v in metadata.items():
            idx_to_ignore = len('npc_dota_hero_')
            vocab.update({int(k) + len(special_tks): v['name'][idx_to_ignore:]})

        return vocab

        
