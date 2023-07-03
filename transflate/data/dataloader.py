from transflate.data.token import tokenize
from transflate.data.Batch import collate_batch

import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset

from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en,
                     batch_size=12000, max_padding=128, is_distributed=False, train=True):
    """
    load Multi30k DE-ENG dataset
    convert it from iterable to map
    prepare DataLoader
        + apply collate_batch : <s></s> + padding-> (ex.128) + stack
    return train_dataloader, valid_dataloader : torch..DataLoader

    train : bool
        True - training AND valid! dataset with batches
        False - only valid dataset

    output: train_dataloader
    ( [0, 14, 38 ... 232, 1, 2, 2, 2, 2], [0, 6, 39, .., 13, 1, 2, 2, 2])
    ( [0, 56, 12, ..., 8, 1, 2, 2, 2, 2], [0, 6, 12, ..., 4, 1, 2, 2, 2])
    ....
    """

    def tokenize_de(text):
        return tokenize(text, spacy_de)
    
    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch=batch,
            src_pipeline=tokenize_de,
            tgt_pipeline=tokenize_en,
            src_vocab=vocab_src,
            tgt_vocab=vocab_tgt,
            device=device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    # dataset from iterable to map. Why? To allows re-call multiple times dataset.
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    valid_dataloader = DataLoader(
    valid_iter_map,
    batch_size=batch_size,
    shuffle=(valid_sampler is None),
    sampler=valid_sampler,
    collate_fn=collate_fn,
    )
    if not train:
        return valid_dataloader

    # dataset from iterable to map. Why? To allows re-call multiple times dataset.
    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
    train_iter_map,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


    
    

