import json
import numpy as np
# import torchtext
from torchtext.vocab import Vectors
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import nltk
nltk.download('punkt')

def read_data(train_dataset: str, valid_dataset: str):
    with open(train_dataset, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(valid_dataset, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    return train_data, val_data

# def clean_text(text):
#     text = re.sub(r"<.*?>", " ", text)
#     text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
#     text = re.sub(r"\s{2,}", " ", text)
#     return text.strip().lower()

def text_to_seq(dataset, word2idx, unk_idx):
    seqs = []
    for sentence in tqdm(dataset):
        seqs.append([word2idx[word] if word in word2idx else unk_idx for word in nltk.word_tokenize(sentence)])
    return seqs

def pad_and_truncate_seqs(seqs, max_seq_len, pad_idx, sos_idx, eos_idx):
    seq_pads = np.zeros((len(seqs), max_seq_len))
    for i, seq in tqdm(enumerate(seqs)):
        pad_len = max_seq_len - len(seq) - 2
        if pad_len > 0:
            seq_pads[i] = np.pad([sos_idx] + seq + [eos_idx], (0, pad_len), 'constant', constant_values=(pad_idx))
        else:
            seq_pads[i] =  [sos_idx] + seq[:max_seq_len - 2] + [eos_idx]
    return seq_pads

def create_data_loader(train_dataset, valid_dataset, batch_size, device,
                       origin_max_len=39,
                       mask_max_len=15,
                       fasttext_vec='./data/fasttext.vec',
                       fasttext_mask_vec='./data/fasttext_mask.vec',
                       stm_vec='./data/stm.vec',
                       stm_mask_vec='./data/stm_mask.vec'):

    print('reading datasets...')
    train_data, val_data = read_data(train_dataset, valid_dataset)

    # load word embedding from *.vec
    print('loading word embedding..')
    fasttext_word_embedding = Vectors(fasttext_vec, cache='./')
    fasttext_mask_word_embedding = Vectors(fasttext_mask_vec, cache='./')
    sentiment_word_embedding = Vectors(stm_vec, cache='./')
    sentiment_mask_word_embedding = Vectors(stm_mask_vec, cache='./')

    print('creating vocab...')
    w2idx = fasttext_word_embedding.stoi
    mask_w2idx = fasttext_mask_word_embedding.stoi
    sos_idx, eos_idx, pad_idx, unk_idx = w2idx['<sos>'], w2idx['<eos>'], w2idx['<pad>'], w2idx['<unk>']

    print('convert text to sequence...')
    origin_train = text_to_seq(train_data['origin'], w2idx, unk_idx)
    restr_train = text_to_seq(train_data['resconstruct'], w2idx, unk_idx)
    mask_train = text_to_seq(train_data['mask'], mask_w2idx, unk_idx)

    origin_val = text_to_seq(val_data['origin'], w2idx, unk_idx)
    restr_val = text_to_seq(val_data['resconstruct'], w2idx, unk_idx)
    mask_val = text_to_seq(val_data['mask'], mask_w2idx, unk_idx)

    print('padding and truncating seq...')
    origin_train_pad = pad_and_truncate_seqs(origin_train, origin_max_len, pad_idx, sos_idx, eos_idx)
    restr_train_pad = pad_and_truncate_seqs(restr_train, origin_max_len, pad_idx, sos_idx, eos_idx)
    mask_train_pad = pad_and_truncate_seqs(mask_train, mask_max_len, pad_idx, sos_idx, eos_idx)

    origin_val_pad = pad_and_truncate_seqs(origin_val, origin_max_len, pad_idx, sos_idx, eos_idx)
    restr_val_pad = pad_and_truncate_seqs(restr_val, origin_max_len, pad_idx, sos_idx, eos_idx)
    mask_val_pad = pad_and_truncate_seqs(mask_val, mask_max_len, pad_idx, sos_idx, eos_idx)
    
    train_tensor = TensorDataset(torch.tensor(origin_train_pad, dtype=torch.long).to(device), torch.tensor(restr_train_pad, dtype=torch.long).to(device), torch.tensor(mask_train_pad, dtype=torch.long).to(device))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    val_tensor = TensorDataset(torch.tensor(origin_val_pad, dtype=torch.long).to(device), torch.tensor(restr_val_pad, dtype=torch.long).to(device), torch.tensor(mask_val_pad, dtype=torch.long).to(device))
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, w2idx, mask_w2idx, fasttext_word_embedding, fasttext_mask_word_embedding, sentiment_word_embedding, sentiment_mask_word_embedding