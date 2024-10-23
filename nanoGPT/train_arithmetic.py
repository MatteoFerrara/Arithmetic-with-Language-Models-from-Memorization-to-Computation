""" Simplified training for arithmetics """

import os
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from model import GPTConfig, GPT
from ArithmeticData import create_dataset,random_split_dataset,shuffle_in_unison,token_split_dataset,value_split_dataset

# --- FUNCTIONS ---

def results_accuracy(preds_idx, target, sequences):
    num_token_correct = torch.sum(preds_idx == target)
    token_accuracy = num_token_correct.item() / len(target)
    seq_preds_idx = preds_idx.view(sequences, -1)
    seq_targets = target.view(sequences, -1)
    seq_len = seq_targets.size(1)
    seq_token_correct = torch.sum(seq_preds_idx == seq_targets, 1) 
    seq_accuracy = (seq_token_correct == seq_len).sum() / sequences
    return token_accuracy, seq_accuracy

def eval_accuracy_autoregressive(data, eval_batch_size,max_in_seq_length,max_out_seq_length,model,device):
    data_size = data.size(0)
    assert(data_size % eval_batch_size == 0)
    iterations = data_size // eval_batch_size
    token_accuracy = 0
    seq_accuracy = 0

    prompt = torch.full((eval_batch_size, max_in_seq_length+1), 0)   # token 0 - unused
    prompt = prompt.to(device)
    
    for iter in range(iterations):
        prompt[:,:max_in_seq_length+1] = data[iter*eval_batch_size : (iter+1)*eval_batch_size,:max_in_seq_length+1]    # prefix + <start of sequence>
        generated = model.generate(prompt, max_out_seq_length-1, temperature=1.0, top_k=1)
        output = generated[:,max_in_seq_length+1:].contiguous().view(-1)
        target = data[iter*eval_batch_size : (iter+1)*eval_batch_size,max_in_seq_length+1:].contiguous().view(-1)
        it_token_accuracy, it_seq_accuracy = results_accuracy(output, target, eval_batch_size)
        token_accuracy += it_token_accuracy
        seq_accuracy += it_seq_accuracy
    return token_accuracy/iterations, seq_accuracy/iterations

# learning rate decay scheduler (cosine with warmup)
def get_lr(it,learning_rate,warmup_iters,lr_decay_iters,min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# ---


def nanogpt_training(op,run_count,epochs,mb_size=128,revert_bit=True,val_set_type='Rnd',out_folder_path=None):
    
    # Data params
    vocab_size = 5  
    # ---

    # baby GPT model
    n_layer = 6
    n_head = 8
    n_embd = 64
    dropout = 0.1
    bias = False # do we use bias inside LayerNorm and Linear layers?
    
    # Exp params
    init_seed=1337
    use_cuda = True  # switch to False to use CPU
    #---

    # optimizer
    learning_rate = 1e-3 # with baby networks can afford to go a bit higher # 1e-3 ?
    # learning rate decay settings
    decay_lr = False # whether to decay the learning rate
    lr_decay_iters = 2000 # make equal to max_iters usually
    warmup_iters = 100 # not super necessary potentially
    min_lr = 1e-4 # learning_rate / 10 usually

    beta1 = 0.9
    beta2 = 0.98 # make a bit bigger because number of tokens per iter is small  # 0.99 ?
    weight_decay = 1e-1
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

    # system
    # bfloat16 is not supported on Windows yet, float16 is just 10% faster than float32 (on my GPU)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    
    print("PyTorch Version:", torch.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda: 
        print("Using Cuda device",torch.cuda.current_device()) 
        print(torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(init_seed)

    device_type = 'cuda' if use_cuda else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # amp is automatic mixed precisio package for PyTorch: see https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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

    # concatenate src_data_train and tgt_data_train in a single np array along axis 1
    train_data = np.concatenate((src_data_train, tgt_data_train), axis=1)
    val_data = np.concatenate((src_data_val, tgt_data_val), axis=1)

    # move datasets to device
    train_data = torch.from_numpy(train_data).to(device)
    val_data = torch.from_numpy(val_data).to(device)

    train_size = train_data.size(0)
    iterations = train_size // mb_size

    run_loss=np.empty((run_count,epochs))
    run_train_seq_acc=np.empty((run_count,epochs))
    run_val_seq_acc=np.empty((run_count,epochs))

    for run in range(run_count):
        run_name ='run_{0}'.format(run+1) 
        
        # model init (from scratch)
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=max_in_seq_length + max_out_seq_length,
                        bias=bias, vocab_size=None, dropout=dropout)   # start with model_args from command line
        # determine the vocab size we'll use for from-scratch training
        model_args['vocab_size'] = vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        checkpoint = None # free up memory

        # compile the model
        if compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0 (not yet for Windows)

        print(f"RUN: {run+1}")
        for epoch in range(epochs):
            ep_train_data = shuffle_in_unison((train_data,))[0]
            ep_loss=0

            model.train()
            for iter in range(iterations):
                # current batch
                it_data = ep_train_data[iter*mb_size : (iter+1)*mb_size]
                X = it_data[:,:-1]
                Y = it_data[:,1:].to(torch.long)
                Y[:,:max_in_seq_length] = -1  # the prefix part is not a target (and does not contribute to loss). 
                                    # if prefix is not used, very slow convergence (100 vs 7 epochs per +)
                
                # determine and set the learning rate for this iteration
                global_iter = epoch * iterations + iter
                lr = get_lr(global_iter,learning_rate,warmup_iters,lr_decay_iters,min_lr) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # forward backward update and using the GradScaler if data type is float16
                # under autocast only the forward pass; backward pass uses the same time defined in forward pass
                with ctx:
                    logits, loss = model(X, Y)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                # clip the gradient
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                ep_loss+=loss
                #run_writer.add_scalar("Train_Loss", loss, epoch*iterations+iter)
                print(".",end="")
                #print("Train Loss:", loss)

            ep_loss/=iterations

            model.eval()
            with ctx:
                train_token_acc, train_seq_acc = eval_accuracy_autoregressive(ep_train_data, 1024,max_in_seq_length,max_out_seq_length,model,device)
                token_val_acc, seq_val_acc = eval_accuracy_autoregressive(val_data, 1024,max_in_seq_length,max_out_seq_length,model,device)

            run_loss[run,epoch]=ep_loss
            run_train_seq_acc[run,epoch]=train_seq_acc
            run_val_seq_acc[run,epoch]=seq_val_acc

            print("")
            print(f"Run {run+1} Epoch {epoch+1}, Train Loss {ep_loss:5.4f}")
            print(f"TRAIN Token% {train_token_acc*100:4.2f}, Sequence% {train_seq_acc*100:4.2f}")
            print(f"VALID Token% {token_val_acc*100:4.2f}, Sequence% {seq_val_acc*100:4.2f}")

        if out_folder_path is not None and os.path.exists(out_folder_path):
            checkpoint = { 'model': model.state_dict(),
                        'model_args': model_args 
                        }
            torch.save(checkpoint, os.path.join(out_folder_path,run_name))
    
    return run_train_seq_acc.mean(axis=0),run_val_seq_acc.mean(axis=0)

