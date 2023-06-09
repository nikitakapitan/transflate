"""
This function is a fancy capsula
The core function is transformers.training.train_worker
"""
from transflate.training.train_worker import train_worker

def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config, architecture):
    """
    config : dict 
    architecture : dict
    """
    return train_worker(
        gpu=0,
        ngpus_per_node=1,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        config=config,
        architecture=architecture,
        is_distributed=False,
    )


   
