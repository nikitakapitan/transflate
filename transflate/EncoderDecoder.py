"""
The goal of EncoderDecoder class is to implement general concept of Encoder + Decoder

input variables:
encoder - any class of encoders
decoder - any class of decoders
src_emb - embeddings of source input
tgt_emb - embeddings of target input
generator -
"""
import torch.nn  as nn

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_emb, tgt_emb, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """supposed to be called in __main__
        src (tensor) : source input text
        tgt (tensor) : self-dynamic output
        """
        memory = self.encode(src, src_mask)
        decoding = self.decode(memory, src_mask, tgt, tgt_mask) 
        return decoding
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)
     
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_emb(tgt), memory, src_mask, tgt_mask)
