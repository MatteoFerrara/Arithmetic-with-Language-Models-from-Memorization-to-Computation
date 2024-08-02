
import torch
import torch.nn as nn
import math

# Code from https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # keep as global matrix (split later)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # the batch is a 3D tensor
        batch_size, seq_length, d_model = x.size()
        # reshape -> batch_size, self.num_heads, seq_length, self.d_k
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        # reshape back 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) if num_heads>0 else None
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff) if d_ff>0 else None
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask) if self.self_attn is not None else None
        x = self.norm1(x + self.dropout(attn_output)) if attn_output is not None else self.norm1(x)
        if self.feed_forward is not None:
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) if num_heads>0 else None
        self.cross_attn = MultiHeadAttention(d_model, num_heads) if num_heads>0 else None
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff) if d_ff>0 else None
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask) if self.self_attn is not None else None
        x = self.norm1(x + self.dropout(attn_output)) if attn_output is not None else self.norm1(x)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask) if self.cross_attn is not None else x
        x = self.norm2(x + self.dropout(attn_output))
        if self.feed_forward is not None:
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, pos_emb,num_heads, num_enc_layers,num_dec_layers, d_ff, max_in_seq_length, max_out_seq_length, dropout):
        super(Transformer, self).__init__()
        self.max_out_seq_length = max_out_seq_length
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.enc_positional_encoding = PositionalEncoding(d_model, max_in_seq_length) if pos_emb else None
        self.dec_positional_encoding = PositionalEncoding(d_model, max_out_seq_length) if pos_emb else None

        if num_enc_layers>0:
            self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_enc_layers)])
        else:
            self.encoder_layers =None
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_dec_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # moved here for pre-computation and on-device allocation (register to move automatically on device)
        self.register_buffer("nopeak_mask", (1 - torch.triu(torch.ones(1, max_out_seq_length-1, max_out_seq_length-1), diagonal=1)).bool())


    def generate_mask(self, tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)   # tgt_mask (64, 1, 99, 1)
        nopeak_mask = self.nopeak_mask
        tgt_mask = tgt_mask & nopeak_mask    #  (64, 1, 99, 99)
        return tgt_mask

    def forward(self, src, tgt):
        tgt_mask = self.generate_mask(tgt)

        src_embedded = self.encoder_embedding(src) 
        tgt_embedded = self.decoder_embedding(tgt)

        if self.enc_positional_encoding is not None:
            src_embedded=self.enc_positional_encoding(src_embedded)

        if self.dec_positional_encoding is not None:
            tgt_embedded=self.dec_positional_encoding(tgt_embedded)

        src_embedded = self.dropout(src_embedded) 
        tgt_embedded = self.dropout(tgt_embedded) 

        enc_output = src_embedded

        if self.encoder_layers is not None:
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, None)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, None, tgt_mask)

        output = self.fc(dec_output)
        return output
    
    # Extend the Class with a new method to perform an iterative generation (autoregressive) of full lenght
#   <out> must be initialized by the caller with a <start of sequence> followed by 'unused' tokens
    def autoregressive_forward(self, src, out):
        for i in range(self.max_out_seq_length-1):
            logits = self.forward(src, out[:,:-1])  
            i_token = logits[:,i].argmax(dim=1)      # the token with highest probability in position i for each pattern in the minibatch
            out[:,i+1] = i_token                     # copy to out in order to use it as input from below

    def forward_with_ingection(self, src, out, dec_layer_embeddings,dec_layer_idx):
        logits = self._forward_with_ingection(src, out[:,:-1], dec_layer_embeddings,dec_layer_idx)  
        tokens = logits.argmax(dim=2)      # the token with highest probability in position i for each pattern in the minibatch
        out[:,1:] = tokens                     # copy to out in order to use it as input from below

    def _forward_with_ingection(self, src, tgt_for_mask, dec_layer_embeddings,dec_layer_idx):
        tgt_mask = self.generate_mask(tgt_for_mask)
        src_embedded = self.dropout(self.enc_positional_encoding(self.encoder_embedding(src)))  # (64, 100, 512)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)
        
        dec_output = dec_layer_embeddings
        for dec_layer in self.decoder_layers[dec_layer_idx+1:]:
            dec_output = dec_layer(dec_output, enc_output, None, tgt_mask)

        output = self.fc(dec_output)
        return output

    def forward_extract_embeddings(self, src, tgt, enc_embeddings, dec_embeddings):
        tgt_mask = self.generate_mask(tgt)
        src_embedded = self.dropout(self.enc_positional_encoding(self.encoder_embedding(src)))  # (64, 100, 512)
        tgt_embedded = self.dropout(self.dec_positional_encoding(self.decoder_embedding(tgt)))  # (64, 99, 512)

        enc_output = src_embedded
        i = 0
        for i, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(enc_output, None)
            if enc_embeddings is not None:
                enc_embeddings[i] = enc_output

        dec_output = tgt_embedded
        for i, dec_layer in enumerate(self.decoder_layers):
            dec_output = dec_layer(dec_output, enc_output, None, tgt_mask)
            if dec_embeddings is not None:
                dec_embeddings[i] = dec_output
            
        output = self.fc(dec_output)
        return output

    def autoregressive_forward_extract_embeddings(self, src, out, enc_embeddings, dec_embeddings):
        for i in range(self.max_out_seq_length-1):
            logits = self.forward_extract_embeddings(src, out[:,:-1], enc_embeddings, dec_embeddings)  
            i_token = logits[:,i].argmax(dim=1)      # the token with highest probability in position i for each pattern in the minibatch
            out[:,i+1] = i_token                     # copy to out in order to use it as input from below
        return logits

