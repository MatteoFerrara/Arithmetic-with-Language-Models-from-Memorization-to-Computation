import torch
import numpy as np
import pandas as pd
import random

def shuffle_in_unison(dataset,init_seed=None):
    new_dataset = []
    len_dataset = dataset[0].size(0)
    if init_seed is not None:
        torch.manual_seed(init_seed)
    rand_indx = torch.randperm(len_dataset)
    for x in dataset:
        new_dataset.append(x[rand_indx])
    return new_dataset

def create_dataset(op,max_num=128,revert_bit=True):
    #src_vocab_size = 5  
    #tgt_vocab_size = 5
    # 0: unused
    # 1: <start>
    # 2: '+' 'x' 'R'
    # 3: '0'
    # 4: '1'

    assert(op == '+' or op == 'x' or op == 'R')
    assert(max_num == 16 or max_num == 128 or max_num == 256)

    if max_num == 256:  
        input_length = 8
        if op == '+' or op == 'R': output_length = 9    # max 255+255 = 510 -> 9 bit 
        elif op == 'x': output_length = 16  # max 255*255 = 65025 -> 16 bit
    elif max_num == 128: 
        input_length = 7
        if op == '+' or op == 'R': output_length = 8     # max 127+127 = 254 -> 8 bit
        elif op == 'x': output_length = 14  # max 127 * 127 = 16129 -> 14 bit
    elif max_num == 16:  
        input_length = 7   # padding before
        if op == '+' or op == 'R': output_length = 5     # max 15+15 = 30 -> 5 bit
        elif op == 'x': output_length = 8   # max 15*15 = 225 -> 8 bit
    
    # sequence format to generate:
    max_in_seq_length = 2 * input_length + 1
    max_out_seq_length = output_length + 1
    # input = A0 A1 A2 A3 A4 A5 A6 <op> B0 B1 B2 B3 B4 B5 B6   (for single_input_length = 7)
    # output = <start> S0 S1 ... S7   (max_out_seq_length = 9)
    tset_input_str = []
    tset_output_str = []
    tset_input_values = []
    tset_output_values = []
    
    idx = 0
    for a in range(max_num):
        for b in range(max_num):
            if op == '+': out_val = a+b
            elif op == 'x': out_val = a*b
            elif op == 'R': out_val = random.randint(0,(max_num-1)+(max_num-1))

            tset_input_values.append([a,b])
            tset_output_values.append(out_val)

            if revert_bit:
                tset_input_str.append('{o1:0{width}b}'.format(o1=a, width=input_length)[::-1]+op+'{o2:0{width}b}'.format(o2=b, width=input_length)[::-1]) 
                tset_output_str.append('s'+'{v:0{width}b}'.format(v=out_val, width=output_length)[::-1])
            else:
                tset_input_str.append('{o1:0{width}b}'.format(o1=a, width=input_length)[:]+op+'{o2:0{width}b}'.format(o2=b, width=input_length)[:]) 
                tset_output_str.append('s'+'{v:0{width}b}'.format(v=out_val, width=output_length)[:])

    assert(len(tset_input_str[0]) == max_in_seq_length)        
    assert(len(tset_output_str[0]) == max_out_seq_length) 

    src_data = torch.full((len(tset_input_str), max_in_seq_length), 3)   # token '0'
    tgt_data = torch.full((len(tset_output_str), max_out_seq_length), 3)   # token '0'
    src_data[:,input_length] = 2   # token '+' or '*'
    tgt_data[:,0] = 1   # token <start>

    # replace tokens '1' 
    for s in range(len(tset_input_str)):
        idx = np.where(np.array(list(tset_input_str[s])) == '1')
        src_data[s,idx]=4
        idx = np.where(np.array(list(tset_output_str[s])) == '1')
        tgt_data[s,idx]=4

    return src_data, tgt_data,tset_input_str,tset_output_str,tset_input_values,tset_output_values

def random_split_dataset(src_data,tgt_data,validation_perc=0.25,init_seed=None):
    dataset_size=src_data.size(0)
    validation_size = int(dataset_size * validation_perc)
    training_size = dataset_size - validation_size
    src_data, tgt_data = shuffle_in_unison((src_data, tgt_data),init_seed)
    src_data_train = src_data[0:training_size]
    tgt_data_train = tgt_data[0:training_size]
    src_data_val = src_data[training_size:]
    tgt_data_val = tgt_data[training_size:]

    return src_data_train, tgt_data_train, src_data_val, tgt_data_val

def token_split_dataset(src_data,tgt_data,tset_input_str):
    nn4096 = list(pd.read_csv('1010101+0101010=01111111_in.txt', delimiter='\t', nrows=4096, index_col = None, header = None).iloc[:,0])
    
    idx_in_train = []
    idx_in_val = []
    for idx in range(len(tset_input_str)):
        if tset_input_str[idx].replace('x','+') in nn4096:
            idx_in_val.append(idx)
        else:
            idx_in_train.append(idx)

    src_data_train = src_data[idx_in_train]
    tgt_data_train = tgt_data[idx_in_train]
    src_data_val = src_data[idx_in_val]
    tgt_data_val = tgt_data[idx_in_val]

    return src_data_train, tgt_data_train, src_data_val, tgt_data_val

def value_split_dataset(src_data,tgt_data,tset_input_values):
    idx_in_train = []
    idx_in_val = []
    for idx in range(len(tset_input_values)):
        [a,b]=tset_input_values[idx]
        if (a>=32 and a<96) and (b>=32 and b<96):
            idx_in_val.append(idx)
        else:
            idx_in_train.append(idx)

    src_data_train = src_data[idx_in_train]
    tgt_data_train = tgt_data[idx_in_train]
    src_data_val = src_data[idx_in_val]
    tgt_data_val = tgt_data[idx_in_val]

    return src_data_train, tgt_data_train, src_data_val, tgt_data_val