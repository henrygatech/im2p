import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

    
    
class SentenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, topic_size, num_layers=1):
        super(SentenceRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=2)
        self.fc1 = nn.Linear(hidden_size, topic_size)
        self.fc2 = nn.Linear(topic_size, topic_size)
        self.relu = nn.ReLU()
        
    def forward(self, features, states = None):
        inputs = features.unsqueeze(1)
        hiddens, states = self.lstm(inputs,states)
        print('hiddens',hiddens)
        print('states',states)
        outputs = self.linear(hiddens)
        probs = self.softmax(outputs)
        print('pred prob',probs)
        top_out_1 = self.relu(self.fc1(hiddens))
        topic = self.relu(self.fc2(top_out_1))
        return probs, topic, states

    def sample(self, features, states = None):
        inputs = features.unsqueeze(1)
        hiddens, states = self.lstm(inputs, states)
        outputs = self.linear(hiddens)
        probs = self.softmax(outputs)
        top_out_1 = self.relu(self.fc1(hiddens))
        topic = self.relu(self.fc2(top_out_1))
        return probs, topic, states
    
        
        '''
        inputs = features.unsqueeze(1)
        predicted_probs = []
        prob = 1
        states = None
        while prob>0.5:
            # Running through the LSTM layer
            lstm_out, states = self.lstm(inputs, states)

            # Running through the linear layer
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            prob = torch.nn.functional.softmax(outputs)
            print(prob)
            predicted_probs.append(prob)
            
            # Updating the input
            inputs = lstm_out.unsqueeze(1)
            
        return predicted_probs
        '''
      
        
        

class WordRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(WordRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, topics, captions):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        inputs = torch.cat((topics, embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs[:,-1]

    def sample(self, inputs, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Initializing and empty list for predictions
        sampled_ids = np.zeros((np.shape(inputs)[0], max_len))
        states = None
        # iterating max_len times
        for index in range(max_len):
            # Running through the LSTM layer
            print('inputs',inputs)
            lstm_out, states = self.lstm(inputs, states)
            #print("weight",list(self.lstm.parameters()))
            print('lstm_out ',lstm_out)
            print('lstm_out shape',lstm_out.shape)
            # Running through the linear layer
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            print('outputs shape',outputs.shape)
            print('outputs ',outputs)
            # Getting the maximum probabilities
            target = outputs.max(1)[1]
            # Appending the result into a list
            sampled_ids[:, index] = target.cpu() 
            
            # Updating the input
            inputs = self.embed(target).unsqueeze(1)
            
        return sampled_ids

        