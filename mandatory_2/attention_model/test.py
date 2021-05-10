import torch

import numpy as np
import torch.nn.functional as F

from torch import nn


##########################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network

        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputlayer : An instance of nn.Linear, shape[number_of_cnn_features, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputlayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config                 = config
        self.vocabulary_size        = config['vocabulary_size']
        self.embedding_size         = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_size     = config['hidden_state_size']
        self.num_rnn_layers         = config['num_rnn_layers']
        self.cell_type              = config['cellType']

        # ToDo
        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        # TODO 1 linear layer, with what output size?
        self.outputlayer = nn.Linear(self.hidden_state_size, self.vocabulary_size)
        # the output size for the image features after the processing via
        # self.inputLayer
        self.nnmapsize = 512
        # TODO
        # Shape (batchsize, 10, 2048)
        # inputlayer will be a sequence of Dropout with a drop probability of
        # 0.25, then a 1x1 2d-convolution with output channel size equal to
        # self.nnmapsize,then a 1d BatchNorm, then a leaky relu
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #           dilation=1, groups=1, bias=True, padding_mode='zeros')
        # nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True,
        #                track_running_stats=True)

        self.inputlayer = nn.Sequential(nn.Dropout(p=0.25),
                                        nn.Conv1d(in_channels=self.number_of_cnn_features,
                                                  out_channels=self.nnmapsize,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0),
                                        nn.BatchNorm1d(self.nnmapsize),
                                        nn.LeakyReLU(negative_slope=0.01))

        self.simplifiedrnn = False

        if True == self.simplifiedrnn:
            if self.cell_type != 'RNN':
                print('unsupported combi: True == self.simplifiedrnn and self.cell_type other than RNN',
                      self.cell_type)
                exit()

            if self.config['num_rnn_layers'] != 1:
                print('unsupported combi: True == self.simplifiedrnn and self.config[num_rnn_layers] !=1',
                      self.config['num_rnn_layers'])
                exit()

            self.rnn = RNN_onelayer_simplified(input_size=self.embedding_size + self.nnmapsize,
                                               hidden_state_size=self.hidden_state_size)

        else:
            self.rnn = RNN(input_size=self.embedding_size + self.nnmapsize,
                           hidden_state_size=self.hidden_state_size,
                           last_layer_state_size=self.hidden_state_size+10,
                           num_rnn_layers=self.num_rnn_layers,
                           cell_type=self.cell_type)


        self.attentionlayer = AttentionLayer(self.hidden_state_size)

        return

    def forward(self,
                cnn_features,
                xTokens,
                is_train,
                current_hidden_state=None):
        """
        Args:
            cnn_features        : Features from the CNN network, shape[batch_size, number_of_cnn_features]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.

        # print(cnn_features.shape)
        # torch.Size([128, 10, 2048])
        imgfeat_processed = self.inputlayer(cnn_features.transpose(1, 2))

        # print("***************Size of imgfeat*****************")
        # print(imgfeat_processed.size())
        # print("**********Should be (128, 10, 512)*************")
        if current_hidden_state is None:
            # TODO
            # initialize initial_hidden_state=  with correct dims, depends on
            # cellyupe
            batch_size = xTokens.shape[0]

            if self.cell_type == "LSTM":
                initial_hidden_state = torch.zeros(self.num_rnn_layers,
                                                   batch_size,
                                                   2*self.hidden_state_sizes,
                                                   device=xTokens.device)

            else:
                raise ValueError("Unsupported cell type:", self.cell_type)

        else:
            initial_hidden_state = current_hidden_state

        # use self.rnn to calculate "logits" and "current_hidden_state"
        logits, current_hidden_state_out = self.rnn(xTokens,
                                                    imgfeat_processed,
                                                    initial_hidden_state,
                                                    self.outputlayer,
                                                    self.attentionlayer,
                                                    self.Embedding,
                                                    is_train)

        return logits, current_hidden_state_out

##########################################################################

class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_state_size,
                 last_layer_state_size,
                 num_rnn_layers,
                 cell_type='LSTM'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers)
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells

        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.last_layer_state_size = last_layer_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        # TODO
        input_size_list = []
        input_size_list.append(input_size)
        #input_size_list.append(2*self.hidden_state_size)
        input_size_list.append(last_layer_state_size)
        # input_size_list should have a length equal to the number of layers
        # and input_size_list[i] should contain the input size for layer i

       # TODO
        # Your task is to create a list (self.cells) of type "nn.ModuleList"
        # and populated it with cells of type "self.cell_type" - depending on
        # the number of rnn layers
        self.cells = nn.ModuleList()
        for i in range(self.num_rnn_layers):
            if self.cell_type == "LSTM":
                self.cells.append(LSTMCell(hidden_state_size=self.hidden_state_size,
                                           input_size=input_size_list[i]))

            else:
                raise ValueError("Cell type", self.cell_type, "not supported")

        return

    def forward(self,
                xTokens,
                baseimgfeat,
                initial_hidden_state,
                outputlayer,
                attentionlayer,
                Embedding,
                is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train:
            seqLen = xTokens.shape[1]  # truncated_backprop_length
        else:
            seqLen = 40  # Max sequence length to be generated

        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert
        # from pytorch tensor to lists and back again.

        # get input embedding vectors
        # .clone())    #(batch, seq, feature = 300)
        embed_input_vec = Embedding(input=xTokens)
        # print(embed_input_vec.shape)
        # exit()
        # dim: (batch,  feature ) # the first input sequence element
        tokens_vector = embed_input_vec[:, 0, :]

        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to
        # calculate logits
        logits_series = []

        #(batchsize, channels, feature)
        #(128, 512, 2048)

        pooling = nn.MaxPool1d(10, stride=1)
        #current_state = list(torch.unbind(initial_hidden_state, dim=0))
        current_state = initial_hidden_state
        baseimgfeat_pooled = pooling(baseimgfeat)
        # print(baseimgfeat.size())
        # print(tokens_vector.size())
        # print(baseimgfeat_pooled.size())
        for kk in range(seqLen):
            updatedstate = torch.zeros_like(current_state, device=xTokens.device)

            # TODO
            # you need to:
            # create your lvl0input,
            lvl0input = torch.cat((baseimgfeat_pooled.squeeze(dim=-1), tokens_vector), dim=1)
            # update the hidden cell state for every layer with inputs depending on the layer index
            currinput = lvl0input
            updatedstate[0, :] = self.cells[0].forward(currinput,
                                                       current_state[0])

            currinput = torch.cat((updatedstate[0, :, :self.hidden_state_size].clone(),
                                   attentionlayer(updatedstate[0, :])), dim = 1)

            updatedstate[1, :] = self.cells[1].forward(currinput,
                                                       current_state[1])

            # if you are at the last layer, then produce logitskk, tokens , run
            # a             logits_series.append(logitskk), see the simplified
            # rnn for the one layer version

            # last layer hidden state ??
            # llhs = updatedstate[-1, :, self.hidden_state_size:]
            # attention attention-weighted sum
            # atws =

            logitskk = outputlayer(updatedstate[-1, :, self.hidden_state_size:])
            # find the next predicted output element
            tokens = torch.argmax(logitskk, dim=1)
            logits_series.append(logitskk)

            current_state = updatedstate
            if kk < seqLen - 1:
                if is_train:
                    tokens_vector = embed_input_vec[:, kk + 1, :]
                elif is_train == False:
                    tokens_vector = Embedding(tokens)

        # Produce outputs
        logits = torch.stack(logits_series, dim=1)
        #current_state = torch.stack(current_state, dim=0)
        return logits, current_state

##########################################################################

class AttentionLayer(nn.Module):
    def __init__(self, hidden_state_size):
        super(AttentionLayer, self).__init__()
        self.attlayer = nn.Sequential(nn.Dropout(p=0.25),
                                      nn.Linear(2*hidden_state_size, 50),
                                      nn.LeakyReLU(negative_slope=0.01),
                                      nn.Linear(50, 10),
                                      nn.Softmax(dim=-1))

    def forward(self, input):
        return self.attlayer(input)


class LSTMCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(LSTMCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

            note: the actual tensor has 2*hidden_state_size because it contains hiddenstate and memory cell
        Returns:
            self.weight_f ...

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        # TODO:
        self.weight_f = nn.Parameter(torch.randn(input_size + hidden_state_size, hidden_state_size) /
                                               np.sqrt(input_size + hidden_state_size))
        self.bias_f = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_i = nn.Parameter(torch.randn(input_size + hidden_state_size, hidden_state_size) /
                                               np.sqrt(input_size + hidden_state_size))
        self.bias_i = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_meminput = nn.Parameter(torch.randn(input_size + hidden_state_size, hidden_state_size) /
                                               np.sqrt(input_size + hidden_state_size))
        self.bias_meminput = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_o = nn.Parameter(torch.randn(input_size + hidden_state_size, hidden_state_size) /
                                               np.sqrt(input_size + hidden_state_size))
        self.bias_o = nn.Parameter(torch.zeros(1, hidden_state_size))

        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, 2*hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, 2*hidden_state_sizes]

        """
        # TODO:
        # Motivated by Lecture notes
        h = state_old[:, :self.hidden_state_size]
        c = state_old[:, self.hidden_state_size:]

        x2 = torch.cat((x, h), dim=1)

        #print(x.shape)
        #print(state_old.shape)
        #print("x2",x2.shape)
        #print("weights",self.weight_i.shape)

        """if self.weight_i.shape[0] != x2.shape[1]:
            print("weight",self.weight_i.shape)
            print("x2",x2.shape)
            print("x with hidden",x.shape)
            print("hidden state size",self.hidden_state_size)"""

        i = torch.sigmoid(torch.mm(x2, self.weight_i) + self.bias_i)    #input gate

        f = torch.sigmoid(torch.mm(x2, self.weight_f) + self.bias_f)    #forget gate

        o = torch.sigmoid(torch.mm(x2, self.weight_o) + self.bias_o)    #output gate

        c_tilde = torch.tanh(torch.mm(x2, self.weight_meminput) + self.bias_meminput)   #candidate memory cell

        #print("f", f.shape)
        #print("c",c.shape)
        #print("c_tilde",c_tilde.shape)

        c = f*c + i*c_tilde   #memory cell update

        state_new = torch.cat((o*torch.tanh(c), c), dim = 1)
        #print(state_new.shape)
        return state_new


##########################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001  # used to not divide on zero

    logits = logits.view(-1, logits.shape[2])
    yTokens = yTokens.view(-1)
    yWeights = yWeights.view(-1)
    losses = F.cross_entropy(input=logits, target=yTokens, reduction='none')

    sumLoss = (losses * yWeights).sum()
    meanLoss = sumLoss / (yWeights.sum() + eps)

    return sumLoss, meanLoss


# ########################################################################################################################
# if __name__ == '__main__':
#
#     lossDict = {'logits': logits,
#                 'yTokens': yTokens,
#                 'yWeights': yWeights,
#                 'sumLoss': sumLoss,
#                 'meanLoss': meanLoss
#     }
#
#     sumLoss, meanLoss = loss_fn(logits, yTokens, yWeights)
#

