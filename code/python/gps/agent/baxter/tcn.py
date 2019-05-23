import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy as copy
from ipdb import set_trace

VOCAB_SIZE = 2

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x, inplace=True)
        return x

class EmbeddingNet(nn.Module):
    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

class PosNet(EmbeddingNet):
    def __init__(self):
        super(PosNet, self).__init__()
        # Input 1
        self.Conv2d_1a = nn.Conv2d(3, 64, bias=False, kernel_size=10, stride=2)
        self.Conv2d_2a = BatchNormConv2d(64, 32, bias=False, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(32, 32, bias=False, kernel_size=3, stride=1)
        self.Conv2d_4a = BatchNormConv2d(32, 32, bias=False, kernel_size=2, stride=1)

        self.Dense1 = Dense(6 * 6 * 32, 32)
        self.alpha = 10

    def forward(self, input_batch):
        # 128 x 128 x 3
        x = self.Conv2d_1a(input_batch)
        # 60 x 60 x 64
        x = self.Conv2d_2a(x)
        # 58 x 58 x 64
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 29 x 29 x 32
        x = self.Conv2d_3a(x)
        # 27 x 27 x 32
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 13 x 13 x 32
        x = self.Conv2d_4a(x)
        # 12 x 12 x 32
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size()[0], -1)
        # 6 x 6 x 32
        x = self.Dense1(x)
        # 32

        return self.normalize(x) * self.alpha

class DenseClassifier(nn.Module):
    def __init__(self, num_classes, h1=20, h2=30):
        super(DenseClassifier, self).__init__()

        self.Conv2d_1a_3x3 = BatchNormConv2d(100, 10, kernel_size=3, stride=1)
        self.SpatialSoftmax = nn.Softmax2d()
        self.FullyConnected7a = Dense(33 * 33 * 100, VOCAB_SIZE)
        self.FullyConnected7b = Dense(128, VOCAB_SIZE)

        self._Conv2d_1a_3x3 = BatchNormConv2d(100, 10, kernel_size=3, stride=1)
        self._SpatialSoftmax = nn.Softmax2d()
        self._FullyConnected7a = Dense(33 * 33 * 100, VOCAB_SIZE)
        self._FullyConnected7b = Dense(128, VOCAB_SIZE)

    def forward(self, x):
        # x = torch.mean(x, axis=0, keepdims=True)
        # 31 x 31 x 20
        # out1 = self.Conv2d_1a_3x3(x)
        out1 = self.SpatialSoftmax(x)
        out1 = self.FullyConnected7a(out1.view(out1.size()[0], -1))
        # out1 = self.FullyConnected7b(out1)

        # out2 = self._Conv2d_1a_3x3(x)
        out2 = self._SpatialSoftmax(x)
        out2 = self._FullyConnected7a(out2.view(out2.size()[0], -1))
        # out2 = self._FullyConnected7b(out2)

        return out1, out2 
        
class TCNModel(EmbeddingNet):
    def __init__(self, inception):  
        super(TCNModel, self).__init__()
        self.transform_input = True
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Conv2d_6a_3x3 = BatchNormConv2d(288, 100, kernel_size=3, stride=1)
        self.Conv2d_6b_3x3 = BatchNormConv2d(100, 20, kernel_size=3, stride=1)
        self.SpatialSoftmax = nn.Softmax2d()
        self.FullyConnected7a = Dense(31 * 31 * 20, 32)

        self.alpha = 10.0

    def forward(self, x):
        if self.transform_input:
            x = copy(x)
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 33 x 33 x 100
        y = self.Conv2d_6a_3x3(x)
        # 31 x 31 x 20
        x = self.Conv2d_6b_3x3(y)
        # 31 x 31 x 20
        x = self.SpatialSoftmax(x)
        # 32
        x = self.FullyConnected7a(x.view(x.size()[0], -1))

        # Normalize output such that output lives on unit sphere.
        # Multiply by alpha as in https://arxiv.org/pdf/1703.09507.pdf
        return self.normalize(x) * self.alpha, x, y


def define_model(pretrained=True):
    return TCNModel(models.inception_v3(pretrained=pretrained))

class TCNDepthModel(EmbeddingNet):
    def __init__(self, inception):  
        super(TCNDepthModel, self).__init__()
        self.transform_input = True
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Conv2d_6a_3x3 = BatchNormConv2d(288, 100, kernel_size=3, stride=1)
        self.Conv2d_6b_3x3 = BatchNormConv2d(100, 20, kernel_size=3, stride=1)
        self.SpatialSoftmax = nn.Softmax2d()
        self.FullyConnected7a = Dense(31 * 31 * 20, 32)

        # Depth layers
        self.Conv2d_depth_1a_3x3 = BatchNormConv2d(1, 64, kernel_size=3, stride=2)
        self.Conv2d_depth_1b_3x3 = BatchNormConv2d(64, 32, kernel_size=3, stride=1)
        self.Conv2d_depth_1c_3x3 = BatchNormConv2d(32, 10, kernel_size=3, stride=1)
        self.SpatialSoftmax_depth = nn.Softmax2d()
        self.FullyConnected2a_depth = Dense(72 * 72 * 10, 10)
        self.alpha = 10.0

    def forward(self, input):
        # Numbers indicate dimensions AFTER passing through layer below the numbers
        x = input[:, :-1] # RGB
        d = input[:, -1] # Depth
        d.unsqueeze_(1)
        if self.transform_input:
            x = copy(x)
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 33 x 33 x 100 
        y = self.Conv2d_6a_3x3(x)
        # 31 x 31 x 20 
        x = self.Conv2d_6b_3x3(y)
        # 31 x 31 x 20 
        x = self.SpatialSoftmax(x)
        # 31 x 31 x 20
        x = self.FullyConnected7a(x.view(x.size()[0], -1))
        # 32

        # Depth
        # 299 x 299 x1
        d = self.Conv2d_depth_1a_3x3(d)
        # 
        d = self.Conv2d_depth_1b_3x3(d)
        # 
        d = self.Conv2d_depth_1c_3x3(d)
        # 145 x 145 x 10
        d = F.max_pool2d(d, kernel_size=3, stride=2)
        # 72 x 72 x 10
        d = self.SpatialSoftmax_depth(d)
        # 72 x 72 x 10
        d = self.FullyConnected2a_depth(d.view(d.size()[0], -1))
        # 10
        out = torch.cat([x, d], 1)
        # 42

        # Normalize output such that output lives on unit sphere.
        # Multiply by alpha as in https://arxiv.org/pdf/1703.09507.pdf
        return self.normalize(out) * self.alpha, out, y


def define_model_depth(pretrained=True):
    return TCNDepthModel(models.inception_v3(pretrained=pretrained))


class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, vocab_size, num_layers, max_seq_length=12):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, feature_size)
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.max_seg_length = max_seq_length
        
    def forward(self, features, lengths):
        """Decode image feature vectors and generates captions."""
        packed = pack_padded_sequence(features, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0][-1])
        outputs = self.sigmoid(outputs)
        # binary cross entropy - model each output as sigmoid
        return outputs

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        inception = models.inception_v3(pretrained=True)
        modules = list(inception.children())[:-1]      # delete the last fc layer.
        self.inception = nn.Sequential(*modules)
        self.linear = nn.Linear(inception.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, shared_features):
        """Extract feature vectors from input images."""
        # with torch.no_grad():
        #     features = self.inception(images)

        features = shared_features.reshape(shared_features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=12):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids