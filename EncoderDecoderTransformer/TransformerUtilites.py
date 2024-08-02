
import torch

def results_accuracy(preds_idx, target, sequences):
    num_token_correct = torch.sum(preds_idx == target)
    token_accuracy = num_token_correct.item() / len(target)
    seq_preds_idx = preds_idx.view(sequences, -1)
    seq_targets = target.view(sequences, -1)
    seq_len = seq_targets.size(1)
    seq_token_correct = torch.sum(seq_preds_idx == seq_targets, 1) 
    seq_accuracy = (seq_token_correct == seq_len).sum() / sequences
    return token_accuracy, seq_accuracy

def eval_accuracy_autoregressive(model, device, src_data, tgt_data, max_out_seq_length, mb_size,revert_bit):
    data_size = src_data.size(0)
    assert(data_size % mb_size == 0)
    iterations = data_size // mb_size
    token_accuracy = 0
    seq_accuracy = 0
    
    out_data = torch.full((mb_size, max_out_seq_length), 0)   # token 0 - unused
    out_data = out_data.to(device)
    out_data[:,0] = 1   # token <start of sequence>

    for iter in range(iterations):
        it_src_data = src_data[iter*mb_size : (iter+1)*mb_size]
        it_tgt_data = tgt_data[iter*mb_size : (iter+1)*mb_size]
        out_data[:,1:] = 0   # reset (except start of sequence)
        model.autoregressive_forward(it_src_data, out_data)
        output = out_data[:, 1:].contiguous().view(-1)
        target = it_tgt_data[:, 1:].contiguous().view(-1)          
        it_token_accuracy, it_seq_accuracy = results_accuracy(output, target, len(out_data))
        token_accuracy += it_token_accuracy
        seq_accuracy += it_seq_accuracy
        
    return token_accuracy/iterations, seq_accuracy/iterations