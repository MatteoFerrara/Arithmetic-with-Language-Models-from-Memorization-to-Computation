
import os
import random
import torch
import numpy as np

from ArithmeticData import create_dataset
from Transformer import Transformer

def hamming(sab, sab2, len):
    ab = sab[:len]
    ab2 = sab2[:len]
    return sum(1 for c1, c2 in zip(ab, ab2) if c1 != c2)

def corr(x,y):
    return np.corrcoef(x,y)[0,1]
    
def transformer_compute_token_and_value_dist(op,model_checkpoint_path):

    # Data params
    src_vocab_size = 5  
    tgt_vocab_size = 5
    revert_bit=True
    # ---

    # Transformer params
    d_model = 64
    pos_emb=True
    num_heads = 8
    num_enc_layers = 6
    num_dec_layers = 6
    d_ff =d_model * 4
    dropout = 0.1
    # ---

    # Exp params
    init_seed=23
    use_cuda = True  # switch to False to use CPU
    #---

    random.seed(init_seed)

    # Inizializzazione PyTorch
    print("PyTorch Version:", torch.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda: 
        print("Using Cuda device",torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(init_seed)

    src_data, tgt_data,_,_,tset_input_values,_=create_dataset(op,revert_bit=revert_bit)
    max_in_seq_length= src_data.size(1)
    max_out_seq_length = tgt_data.size(1)

    print('source size:',src_data.size(), 'target size:',tgt_data.size())

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,pos_emb, num_heads, num_enc_layers,num_dec_layers, d_ff, max_in_seq_length, max_out_seq_length, dropout)

    checkpoint = torch.load(model_checkpoint_path)
    transformer.load_state_dict(checkpoint)
    transformer = transformer.to(device)

    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)

    transformer.eval()

    val_indices=[index for index,[a,b] in enumerate(tset_input_values) if a==b]
    src_data_val=src_data[val_indices]
    tgt_data_val=tgt_data[val_indices]

    samples = len(val_indices)
    distances = (samples*(samples-1))//2 

    t_in_dist = np.zeros(distances)
    v_in_dist = np.zeros(distances)
    t_out_dist = np.zeros(distances)
    v_out_dist = np.zeros(distances)
    lm_enc_dist = np.zeros((num_enc_layers, distances))
    lm_dec_dist = np.zeros((num_dec_layers, distances))
    enc_embeddings = torch.full((num_enc_layers, len(src_data_val), max_in_seq_length, d_model), 0, dtype = torch.float32)
    dec_embeddings = torch.full((num_dec_layers, len(src_data_val), max_out_seq_length-1, d_model), 0, dtype = torch.float32) 
    out_data = torch.full((len(src_data_val), max_out_seq_length), 0)   # token 0 - unused
    enc_embeddings = enc_embeddings.to(device)
    dec_embeddings = dec_embeddings.to(device)
    out_data = out_data.to(device)

    # autoregressive forward
    out_data[:,0] = 1   # token <start of sequence>
    logits = transformer.autoregressive_forward_extract_embeddings(src_data_val, out_data, enc_embeddings, dec_embeddings)

    i = 0
    for ab in range(128):
        assert ab==tset_input_values[val_indices[ab]][0]
        for ab2 in range(ab+1, 128):
            assert ab2==tset_input_values[val_indices[ab2]][0]
            t_in_dist[i] = hamming(src_data_val[ab], src_data_val[ab2], src_data_val.size(1))
            v_in_dist[i] = abs(ab - ab2)
            t_out_dist[i] = hamming(tgt_data_val[ab], tgt_data_val[ab2], tgt_data_val.size(1))
            if op == '+': v_out_dist[i] = abs(2*ab - 2*ab2)
            elif op == 'x': v_out_dist[i] = abs(ab**2 - ab2**2)
            ab_logit_dist = logits[ab].view(-1)
            ab2_logit_dist = logits[ab2].view(-1)
            for layer in range(num_enc_layers):  # encoder
                ab_embedding = enc_embeddings[layer, ab].view(-1)
                ab2_embedding = enc_embeddings[layer, ab2].view(-1)
                lm_enc_dist[layer,i] = (ab_embedding - ab2_embedding).pow(2).sum().sqrt()
            for layer in range(num_dec_layers):  # decoder
                ab_embedding = dec_embeddings[layer, ab].view(-1)
                ab2_embedding = dec_embeddings[layer, ab2].view(-1)
                lm_dec_dist[layer,i] = (ab_embedding - ab2_embedding).pow(2).sum().sqrt()           
            i+=1

    print('INPUT')
    print("%.3f %.3f" % (corr(t_in_dist, t_in_dist), corr(v_in_dist, t_in_dist)))
    print('encoder')
    for layer in range(num_enc_layers):
        print("%.3f %.3f" % (corr(t_in_dist, lm_enc_dist[layer]), corr(v_in_dist, lm_enc_dist[layer])))
    print('decoder')
    for layer in range(num_dec_layers):
        print("%.3f %.3f" % (corr(t_in_dist, lm_dec_dist[layer]), corr(v_in_dist, lm_dec_dist[layer])))

    print()
    print('OUTPUT')
    print("%.3f %.3f" % (corr(t_out_dist, t_out_dist), corr(v_out_dist, t_out_dist)))
    print('encoder')
    for layer in range(num_enc_layers):
        print("%.3f %.3f" % (corr(t_out_dist, lm_enc_dist[layer]), corr(v_out_dist, lm_enc_dist[layer])))
    print('decoder')
    for layer in range(num_dec_layers):
        print("%.3f %.3f" % (corr(t_out_dist, lm_dec_dist[layer]), corr(v_out_dist, lm_dec_dist[layer])))