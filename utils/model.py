import random
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F

"""
- DATA: we split each docs to 2 parts: 75% and 25%
- Model: we formulate this problem as seq2seq and mask prediction task, model consist of 2 components: encoder and decoder
    + Encoder, we use 2 bi-gru for learning local and global features
    + Decoder has 2 parts: the first reconstruct invert 75% docs (.i.e from token `t` to 0), the rest will be mask prediction with 50%
"""


class Encoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 device,
                 weight_top_embedding,
                 bot_embedd_dim=300,
                 num_layers=2,
                 bidirectional=True,
                 p=0.2,
                 batch_first=True):

        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.device = device
        self.top_embedd_layer = nn.Embedding.from_pretrained(embeddings=weight_top_embedding.vectors)
        self.bot_embedd_layer = nn.Embedding(embedding_dim=bot_embedd_dim, num_embeddings=weight_top_embedding.vectors.size(0))
        
        # the first rnn to learning sentiment text, using sentiment word embedding
        self.rnn1 = nn.GRU(input_size=weight_top_embedding.dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)

        # the second rnn to learning global represent of text, using fasttex
        self.rnn2 = nn.GRU(input_size=bot_embedd_dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)

        self.dropout = nn.Dropout(p)

    def _init_hidden(self, batch_size):
        """
        Args:
            batch_size: int
        Returns:
            h (Tensor): the first hidden state of rnn
        """

        if self.bidirectional:
            h = torch.zeros((self.num_layers*2, batch_size, self.hidden_size))
        else:
            h = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        nn.init.xavier_normal_(h)

        return h.to(self.device)

    def forward(self, top_x, bot_x):
        """
        Args:
            top_x, bot_x: Tensor(batch_size, seq_len)
        Returns:
            ouput1, h1: Tuple(Tensor, Tensor)
            output2, h2: Tuple(Tensor, Tensor)
        """

        top_embedded = self.top_embedd_layer(top_x)
        bot_embedded = self.bot_embedd_layer(bot_x)

        h = self._init_hidden(top_x.size(0))

        output1, h1 = self.rnn1(top_embedded, h) # h1, h2 size of (D ∗ num_layers, batch_size, hidden_size)
        output2, h2 = self.rnn2(bot_embedded, h) # output1, output2 size of (batch_size, seq_len, D * hidden_size)

        return (output1, h1), (output2, h2)

class BahdanauAttention(nn.Module):
    def __init__(self, dec_dim: int, enc_dim: int, num_hiddens: int):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, num_hiddens, bias=False)
        self.W2 = nn.Linear(enc_dim, num_hiddens, bias=False)
        self.v = nn.Linear(num_hiddens, 1, False)

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            value (Tensor(batch_size, seq_len, encoder hidden dimension): the hidden_state of tokens in encoder
            query (Tensor(batch_size, 1, decoder hidden dimension)): the hidden state of decoder at time step t
        Returns:
            attention_weight (Tensor)
            context_vector (Tensor)
        """

        score = self.v(torch.tanh(self.W1(query) + self.W2(value))) # `score` size of: (batch_size, seq_len, 1)

        attention_weight = F.softmax(score.squeeze(-1), dim=1) # `attention` size of: (batch_size, seq_len)

        context_vector = torch.bmm(attention_weight.unsqueeze(1), value).squeeze(1) # `context_vector` size of: (batch_size, num_hiddens)
        return attention_weight, context_vector


class Decoder(nn.Module):
    def __init__(self, 
                 hidden_size,
                 weight_top_embedding,
                 encoder_output_dim,
                 bot_embedd_dim=300,
                 num_layers=2,
                 bidirectional=True,
                 p=0.2,
                 batch_first=True):

        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.vocab_size = weight_top_embedding.vectors.size(0)

        self.top_embedd_layer = nn.Embedding.from_pretrained(embeddings=weight_top_embedding.vectors)
        self.bot_embedd_layer = nn.Embedding(num_embeddings=weight_top_embedding.vectors.size(0), embedding_dim=bot_embedd_dim)

        self.rnn1 = nn.GRU(input_size=weight_top_embedding.dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)

        self.rnn2 = nn.GRU(input_size=bot_embedd_dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=p, bidirectional=bidirectional, batch_first=batch_first)
        
        self.dropout = nn.Dropout(p)

        self.attention1 = BahdanauAttention(dec_dim=hidden_size * self.D, enc_dim=encoder_output_dim,
                                            num_hiddens=hidden_size * self.D)
        self.attention2 = BahdanauAttention(dec_dim=hidden_size * self.D, enc_dim=encoder_output_dim,
                                            num_hiddens=hidden_size * self.D)
        
        self.fc = nn.Sequential(nn.Linear(2*(encoder_output_dim + hidden_size*2), 2 * encoder_output_dim),
                                nn.ReLU())

    def forward(self, x1: Tensor, output1_encoder: Tensor, h1_encoder: Tensor, 
                x2: Tensor, output2_encoder: Tensor, h2_encoder: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        """
        Args:
            `x1`, `x2` (Tensor): seq number for embedding layer
            `output1_encoder`, `h1_encoder`, `output2_encoder`, `h2_encoder` (Tensor): output of encoder layer
        """

        # `x1`, `x2` size of batch_size -> (batch_size, 1): `1` denote seq_length
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)

        top_embedd = self.dropout(self.top_embedd_layer(x1))
        bot_embedd = self.dropout(self.bot_embedd_layer(x2)) # (batch_size, 1, embedd dim)

        output1, h1 = self.rnn1(top_embedd, h1_encoder)
        output2, h2 = self.rnn2(bot_embedd, h2_encoder)

        _, context_vector1 = self.attention1(output1, output1_encoder)
        _, context_vector2 = self.attention2(output2, output2_encoder)

        output1 = torch.cat((output1.squeeze(1), context_vector1), dim=1)
        output2 = torch.cat((output2.squeeze(1), context_vector2), dim=1)
        output = torch.cat((output1, output2), dim=1) # [batch_size, 2* (encoder_output_dim + hidden_size*2)]
        output = self.fc(output) # [batch_size, 2 * encoder_output_dim]

        return output, h1, h2

class SentimentModel(nn.Module):
    def __init__(self, encoder, decoder1, decoder2, num_aspect):
        super(SentimentModel, self).__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.relu = nn.ReLU()
        self.T = nn.Linear(2*(encoder.hidden_size*encoder.D), num_aspect)
        self.Z1 = nn.Linear(num_aspect, decoder1.vocab_size)
        self.Z2 = nn.Linear(num_aspect, decoder2.vocab_size)

    def forward(self, top_source, bot_source, top_res, top_mask, teacher_forcing_ratio=0.5):
        """
        Args:
            `top_soucre`, `bot_source` (Tensor): input of encoder block
            `top_res`, `bot_res` (Tensor): input of the first decoder block
            `top_mask`, `bot_mask` (Tensor): input of the second decoder block
        """
        batch_size = top_mask.size(0)
        mask_len = top_mask.size(1)
        res_len = top_res.size(1)

        output_reverts = torch.zeros(res_len, batch_size, self.decoder1.vocab_size)
        output_masks = torch.zeros(mask_len, batch_size, self.decoder2.vocab_size)

        (output1, h1), (output2, h2) = self.encoder(top_source, bot_source)

        input_ = top_res[:, 0]
        for i in range(1, res_len):
            output_revert, h1, h2 = self.decoder1(x1=input_, output1_encoder=output1, h1_encoder=h1, x2=input_, output2_encoder=output2, h2_encoder=h2)

            output_revert = self.relu(self.T(output_revert))
            output_revert = self.Z1(output_revert)

            output_reverts[i] = output_revert
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output_revert.argmax(1)
            input_ = top_res[:, i] if teacher_force else top1

        (output1, h1), (output2, h2) = self.encoder(top_source, bot_source)

        input_ = top_mask[:, 0]
        for i in range(1, mask_len):
            output_mask, h1, h2 = self.decoder2(x1=input_, output1_encoder=output1, h1_encoder=h1, x2=input_, output2_encoder=output2, h2_encoder=h2)

            output_mask = self.relu(self.T(output_mask))
            output_mask = self.Z2(output_mask)

            output_masks[i] = output_mask
            teacher_force1 = random.random() < teacher_forcing_ratio

            top1 = output_mask.argmax(1)
            input_ = top_mask[:, i] if teacher_force1 else top1

        # `output_reverts`, `output_mask` size of : (revert seq_len/mask seq_len, batch_size, decoder1.vocab_size/decoder2.vocab_size)
        return output_reverts, output_masks

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0