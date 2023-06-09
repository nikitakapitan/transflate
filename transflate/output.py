from transflate.data.dataloader import create_dataloaders
import torch
from transflate.main import make_model
from transflate.data.Batch import Batch, collate_batch
from transflate.helper import following_mask

# def translate
from torch.utils.data import DataLoader

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    tgt = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1): # loop over output words (decoded)
        tgt_mask = following_mask(tgt.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, tgt, tgt_mask)

        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=1).unsqueeze(0)
        tgt=torch.cat([tgt, next_word],dim=1)
    return tgt

def check_outputs(valid_dataloader, model, vocab_src, vocab_tgt,
                n_examples=15, pad_idx=2, eos_string="</s>"):
                
    results = [()] * n_examples
    for idx in range(n_examples):
        print(f"\nExample {idx} ======\n")
        b = next(iter(valid_dataloader))
        rb = Batch(src=b[0], tgt=b[1], pad=pad_idx)
        # greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x!=pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x!=pad_idx]
        
        src_tokens = " ".join(src_tokens).replace("\n", "")
        tgt_tokens = " ".join(tgt_tokens).replace("\n", "")

        print(f"Source text (Input) {src_tokens}")
        print(f"Target Text (Ground Truth) {tgt_tokens}")

        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (" ".join([vocab_tgt.get_itos()[x] for x in model_out if x!= pad_idx])\
            .split(eos_string, 1)[0] + eos_string)
        print(f"Model Output {model_txt}")

        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=5):

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
            'dropout' : 0.1
        }

    print('Preparing Data...')
    _, valid_dataloader = create_dataloaders(
    torch.device("cpu"),
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=1,
    is_distributed=False,
    )

    print('Loading Trained model...')
    model = make_model(
    src_vocab_len=architecture['src_vocab_len'],
    tgt_vocab_len=architecture['tgt_vocab_len'],
    N=architecture['N'],
    d_model=architecture['d_model'],
    d_ff=architecture['d_ff'],
    h=architecture['h'],
    dropout=architecture['dropout'],
    )
    
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("checking Model Outputs:")
    example_data = check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)
    return model, example_data

def translate(text, vocab_src, vocab_tgt, spacy_de, spacy_en):

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
            'dropout' : 0.1
        }

    batch_text = [(text, "")]
    tokenize_de = lambda x : [token.text for token in spacy_de.tokenizer(x)]
    tokenize_en = lambda x : [token.text for token in spacy_en.tokenizer(x)]

    collate_fn = lambda x:  collate_batch(
            batch=x,
            src_pipeline=tokenize_de,
            tgt_pipeline=tokenize_en,
            src_vocab=vocab_src,
            tgt_vocab=vocab_tgt,
            device=torch.device("cpu"),
            max_padding=data_setup['max_padding'],
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    text_dataloader = DataLoader(text, collate_fn = collate_fn)

    print('Loading Trained model...')
    model = make_model(
    src_vocab_len=architecture['src_vocab_len'],
    tgt_vocab_len=architecture['tgt_vocab_len'],
    N=architecture['N'],
    d_model=architecture['d_model'],
    d_ff=architecture['d_ff'],
    h=architecture['h'],
    dropout=architecture['dropout'],
    )
    
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )
