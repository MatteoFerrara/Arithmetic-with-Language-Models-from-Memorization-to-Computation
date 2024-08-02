import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

from Transformer import Transformer
from TransformerUtilites import eval_accuracy_autoregressive
from ArithmeticData import create_dataset,random_split_dataset,shuffle_in_unison,token_split_dataset,value_split_dataset

def transformer_training(op,run_count,epochs,mb_size=128,revert_bit=True,val_set_type='Rnd',out_folder_path=None):

    # Data params
    src_vocab_size = 5  
    tgt_vocab_size = 5
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

    print("PyTorch Version:", torch.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda: 
        print("Using Cuda device",torch.cuda.current_device()) 
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(init_seed)

    src_data, tgt_data,tset_input_str,_,tset_input_values,_=create_dataset(op,revert_bit=revert_bit)
    print('dataset size:',src_data.size(0))

    max_in_seq_length= src_data.size(1)
    max_out_seq_length = tgt_data.size(1)

    if val_set_type=='Rnd':
        src_data_train, tgt_data_train, src_data_val, tgt_data_val = random_split_dataset(src_data, tgt_data,validation_perc=0.25)
    elif val_set_type=='VSt':
        src_data_train, tgt_data_train, src_data_val, tgt_data_val = token_split_dataset(src_data, tgt_data, tset_input_str)
    elif val_set_type=='VSv':
        src_data_train, tgt_data_train, src_data_val, tgt_data_val = value_split_dataset(src_data, tgt_data, tset_input_values)

    print('training source size:',src_data_train.size(), 'training target size:',tgt_data_train.size())
    print('val source size:',src_data_val.size(), 'val target size:',tgt_data_val.size())

    train_size = src_data_train.size(0)
    iterations = train_size // mb_size

    src_data_train = src_data_train.to(device)
    tgt_data_train = tgt_data_train.to(device)
    src_data_val = src_data_val.to(device)
    tgt_data_val = tgt_data_val.to(device)

    exp_avg_loss=np.zeros((epochs,))
    exp_avg_train_seq_acc=np.zeros((epochs,))
    exp_avg_val_seq_acc=np.zeros((epochs,))

    for run in range(run_count):
        run_name ='run_{0}'.format(run+1)
        transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,pos_emb, num_heads, num_enc_layers,num_dec_layers, d_ff, max_in_seq_length, max_out_seq_length, dropout)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        # Move to device
        transformer = transformer.to(device)
        
        print(f"RUN: {run+1}")
        for epoch in range(epochs):
            ep_src_data, ep_tgt_data = shuffle_in_unison((src_data_train, tgt_data_train))
            ep_loss=0

            transformer.train()
            for iter in range(iterations):
                it_src_data = ep_src_data[iter*mb_size : (iter+1)*mb_size]
                it_tgt_data = ep_tgt_data[iter*mb_size : (iter+1)*mb_size]

                optimizer.zero_grad()
                output = transformer(it_src_data, it_tgt_data[:, :-1])    
                preds = output.contiguous().view(-1, tgt_vocab_size)    
                target = it_tgt_data[:, 1:].contiguous().view(-1)          
                loss = criterion(preds, target)
                ep_loss+=loss

                loss.backward()
                optimizer.step()
                print(".",end="")
            
            ep_loss/=iterations

            transformer.eval()
            train_token_acc, train_seq_acc = eval_accuracy_autoregressive(transformer, device, ep_src_data, ep_tgt_data, max_out_seq_length, 1024,revert_bit)
            token_val_acc, seq_val_acc = eval_accuracy_autoregressive(transformer, device, src_data_val, tgt_data_val, max_out_seq_length, 1024,revert_bit)

            exp_avg_loss[epoch]+=ep_loss
            exp_avg_train_seq_acc[epoch]+=train_seq_acc
            exp_avg_val_seq_acc[epoch]+=seq_val_acc

            print("")
            print(f"Run {run+1} Epoch {epoch+1}, Train Loss {ep_loss:5.4f}")
            print(f"TRAIN: Token% {train_token_acc*100:4.2f}, Sequence% {train_seq_acc*100:4.2f}")
            print(f"VALID: Token% {token_val_acc*100:4.2f}, Sequence% {seq_val_acc*100:4.2f}")

        if out_folder_path is not None and os.path.exists(out_folder_path):
            torch.save(transformer.state_dict(), os.path.join(out_folder_path,run_name))

    if run_count>1:
        exp_avg_loss/=run_count
        exp_avg_train_seq_acc/=run_count
        exp_avg_val_seq_acc/=run_count  

    return exp_avg_train_seq_acc,exp_avg_val_seq_acc