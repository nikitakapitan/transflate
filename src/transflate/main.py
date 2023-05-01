
import torch.nn as nn
from copy import deepcopy as dcopy
from transflate.MultiHeadedAttention import MultiHeadedAttention
from transflate.PositionWiseFeedForward import PositionWiseFeedForward

from transflate.PositionalEncoding import PositionalEncoding
from transflate.Embeddings import Embeddings

from transflate.EncoderDecoder import EncoderDecoder
from transflate.Encoder import Encoder
from transflate.EncoderLayer import EncoderLayer
from transflate.Decoder import Decoder
from transflate.DecoderLayer import DecoderLayer

from transflate.Generator import Generator

def make_model(src_vocab_len, tgt_vocab_len, N=6, d_model=512, d_ff=2048, h=8, dropout=0.0):
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(
            layer=EncoderLayer(
                size=d_model,
                self_attn=dcopy(attn),
                feed_forward=dcopy(ff),
                dropout=dropout
            ),
            N=N #6
        ),
        decoder=Decoder(
            layer=DecoderLayer(
                size=d_model,
                self_attn=dcopy(attn),
                src_attn=dcopy(attn),
                feed_forward=dcopy(ff),
                dropout=dropout
            ),
            N=N #6
        ),
        src_emb=nn.Sequential(
            Embeddings(
                vocab_len=src_vocab_len,
                d_model=d_model,
                
            ),
            dcopy(position)
        ),
        tgt_emb=nn.Sequential(
            Embeddings(
                vocab_len=tgt_vocab_len,
                d_model=d_model,
            ),
            dcopy(position)
        ),
        generator=Generator(
            d_model=d_model,
            vocab_len=tgt_vocab_len
        )

    )

    # init params with Glorot / fan_avg:
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

if __name__ == "__main__":
    make_model(11,11,2)