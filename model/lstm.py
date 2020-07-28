import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BreakLine(nn.Module):

    def __init__(self, number_of_layer, batch, hidden_size, num_directions=1):
        super(BreakLine, self).__init__()
        bidirection_flag=False
        if num_directions==2:
            bidirection_flag=True
        self.lstm=nn.LSTM(input_size=32, hidden_size= hidden_size,\
             num_layers=number_of_layer , bidirectional=bidirection_flag)
        self.h0= torch.randn(number_of_layer * num_directions, batch, hidden_size).cuda()
        self.c0= torch.randn(number_of_layer * num_directions, batch, hidden_size).cuda()
        self.classification = nn.Linear(hidden_size,1)

    def forward(self,features):
        output, states = self.lstm(features, (self.h0,self.c0))
        #output, states = self.lstm(features)
        output = output.permute(1,0,2)
        # output ==> seq_len, batch, num_directions * hidden_size
        # h_n = states[0]
        # c_n = states[1]
        output = F.sigmoid(self.classification(output).squeeze())  # batchxseq_len
        return output

class BreakLine_v2(nn.Module):

    def __init__(self, hidden_size,output_size=1,feature_feature=32):
        super(BreakLine_v2, self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.feature_feature=feature_feature
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.combine = nn.Linear(self.feature_feature+1, self.hidden_size)
        
    def forward(self,features,hidden, output):
        output = torch.cat((output, features), 1)
        #print(output.shape)
        output = self.combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.sigmoid(self.out(output[0]))
        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

if __name__ == '__main__':
    print('loading model')
    hidden_size=64
    batch_size=8
    lstm_model = BreakLine_v2( hidden_size=hidden_size,output_size=1,feature_feature=32).cuda()
    input = Variable(torch.randn(250, 8, 32)).to(device)
    hidden = lstm_model.initHidden(batch_size)
    output = torch.ones(8, 1, device=device)*(-1)
    for index in range(input.shape[1]):
        output, hidden = lstm_model(input[index,...].cuda(), hidden.cuda(), output.cuda())
        print(output.shape)









