import torch

from transflate.data.token import load_tokenizers
from transflate.data.vocab import load_vocab

from transflate.main import make_model
from transflate.output import check_outputs

from torch.utils.data import DataLoader
from transflate.data.token import tokenize
from transflate.data.Batch import collate_batch

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de=spacy_de, spacy_en=spacy_en)


# translate.google.com
# "Drei Hunde in schwarzen Jacken kaufen Milch in der Innenstadt"
YOUR_GERMAN_SENTENCE = "Der große Junge geht zur Schule und spricht mit Vögeln"

data_setup = {
    'max_padding' : 128,
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

def collate_fn(batch):
        return collate_batch(
            batch=batch,
            src_pipeline=lambda x : tokenize(x, spacy_de),
            tgt_pipeline=lambda x : tokenize(x, spacy_en),
            src_vocab=vocab_src,
            tgt_vocab=vocab_tgt,
            device=None,
            max_padding=data_setup['max_padding'],
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

user_input = DataLoader([(YOUR_GERMAN_SENTENCE, 'None')], collate_fn=collate_fn)

model = make_model(
    src_vocab_len=architecture['src_vocab_len'],
    tgt_vocab_len=architecture['tgt_vocab_len'],
    N=architecture['N'],
    d_model=architecture['d_model'],
    d_ff=architecture['d_ff'],
    h=architecture['h'],
    dropout=architecture['p_dropout'],
    )

model.load_state_dict(
    torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
)

example_data = check_outputs(
        user_input, model, vocab_src, vocab_tgt, n_examples=1
    )