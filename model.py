from ctypes.wintypes import tagSIZE
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from crf import CRF
from typing import Tuple
from train import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class NERModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_to_ix: dict,
        embedding_dim,
        hidden_dim,
        num_laters,
        pre_word_embeds=None,
        use_gpu=False,
        use_crf=False,
        dropout=0.5
    ):
        super(NERModel, self).__init__()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_laters
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.start_tag_id = tag_to_ix[START_TAG]
        self.end_tag_id = tag_to_ix[STOP_TAG]
        self.dropout= nn.Dropout(dropout)

        # self.word_embeds = word_embeds
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pre_word_embeds))
            self.bilstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,num_layers=num_laters,bidirectional=True,batch_first=True)
            self.ffnn = nn.Linear(hidden_dim*2, self.tagset_size)
        if self.use_crf:
            self.crf = CRF(self.tag_to_ix, self.use_gpu, self.start_tag_id, self.end_tag_id)

        """
        
        Here is where you need to implement the initilization of our NER model.
        In this function, you need to initialize the following variables:
        1. self.dropout: The dropout layer.
        2. self.word_embeds: The embedding layer. You need to initialize the embedding layer with the pre-trained word embeddings.
        3. our model part, you need to think about how to get features from BiLSTM model. 
        You also need to think of how to connect the BiLSTM model and the CRF layer.
        
        The number of expected lines is 10 - 20 lines.
        
        """
    # def init_hidden(self):
    #     return (torch.randn(4, 32, self.hidden_dim),
    #             torch.randn(4, 32, self.hidden_dim))

    def _get_features(self, sentence: torch.Tensor):
        """

        This is the function to get the features of the sentences from the BiLSTM model.
        You need to implement this method to get the features of the sentences. 
        You should pass sentences to the BiLSTM model and get the project the output of the BiLSTM model to the tag space through an FFNN layer.
        The number of expected lines is 5 - 10 lines.

        Args:
            sentence (torch.Tensor): The input sentence to be processed. The shape of the tensor is (batch_size, seq_len, embedding_size).

        Returns:
            torch.Tensor: The output of the BiLSTM model.
        """
        # self.hidden = self.init_hidden()
        lstm_out, _ = self.bilstm(sentence)
        lstm_out = self.dropout(lstm_out)
        # lstm_out = lstm_out.view(len(sentence))
        # Apply FFNN to get the features
        lstm_feats = self.ffnn(lstm_out)
        return lstm_feats

    def forward(self, sentence: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """
        This is the function that will be called when the model is being trained.
        You need to implement the forward function of the model here.
        The loss for BiLSTM-CRF model is the negative log likelihood of the model.
        The loss for BiLSTM model is the cross entropy loss.
        The number of expected lines is 5 - 10 lines.

        Args:
            sentence (torch.Tensor): The input sentence to be processed.
            tags (torch.Tensor): The ground truth tags of the input sentence.

        Returns:
            scores (torch.Tensor): The output of the model. It is the loss of the model.
        """
        lstm_feats = self._get_features(self.word_embedding(sentence))
        if self.use_crf:
        # Use CRF loss
            loss = self.crf.forward(lstm_feats, tags)
        else:
        # Use Cross-Entropy loss
            loss = F.cross_entropy(lstm_feats.view(-1, self.tagset_size), tags.view(-1))
        return loss

    def inference(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is the function that will be called when the model is being tested.
        You need to implement the inference function of the model here.
        The number of expected lines is 5 - 10 lines.
        
        Args:
            sentence (torch.Tensor): The input sentence to be processed.

        Returns:
            The score and the predicted tags of the input sentence.
            score (torch.Tensor): The score of the predicted tags.
            tag_seq (torch.Tensor): The predicted tags of the input sentence.
            
        """
        lstm_feats = self._get_features(self.word_embedding(sentence))
        if self.use_crf:
            # Use Viterbi decode for inference
            scores, tag_seq = self.crf.inference(lstm_feats)
        else:
        # Use argmax to get the predicted tags
            scores, tag_seq = torch.max(lstm_feats, dim=-1)
        return scores, tag_seq
