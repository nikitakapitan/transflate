
import torch
from transflate.data.token import load_tokenizers
from transflate.data.vocab import load_vocab
from transflate.training.train import train_model

if not torch.cuda.is_available():
  raise ValueError('change runtime to GPU')


spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de=spacy_de, spacy_en=spacy_en)

train_config = {
        'batch_size' : 32,
        'distributed' : False,
        'num_epochs' : 8,
        'accum_iter' : 10, # nb of gradient accumulation steps
        'base_lr' : 1.0,
        'max_padding' : 72, # add blanks to have total 72 tokens.
        'warmup' : 3000,
        'file_prefix' : 'multi30k_model_',
    }
architecture = {
        'src_vocab_len' : len(vocab_src),
        'tgt_vocab_len' : len(vocab_tgt),
        'N' : 6, # loop
        'd_model' : 512, # emb
        'd_ff' : 2048,
        'h' : 8,
        'p_dropout' : 0.1
    }

model = train_model(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        config=train_config,
        architecture=architecture,
        )