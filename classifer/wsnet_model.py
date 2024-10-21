import torch.nn as nn
import torch.nn.functional as F
#define the CNN architecture
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv1d(1,8,kernel_size=4,stride=2, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4,stride=2)

        self.conv2 = nn.Conv1d(6,10,5,padding=0)


        self.conv3 = nn.Conv1d

        self.pool = nn.MaxPool1d(2, 2)

        self.ff1 = nn.Linear(4*4*10,56)
        self.ff2 = nn.Linear(56,10)

    def forward(self, x):
        # adding sequence of convolutional and max pooling layers
        #input dim-28*28*1
        x = self.conv1(x)# After convolution operation, output dim - 24*24*6
        x = self.pool(x)# After Max pool operation output dim - 12*12*6
        x = self.conv2(x)# After convolution operation  output dim - 8*8*10
        x = self.pool(x)# max pool output dim 4*4*10

        x = x.view(-1,4*4*10) # Reshaping the values to a shape appropriate to the input of fully connected layer
        x = F.relu(self.ff1(x)) # Applying Relu to the output of first layer
        x = F.sigmoid(self.ff2(x)) # Applying sigmoid to the output of second layer
        return x

class CNN2(nn.Module):
    def __init__(self,in_num,out_num):
        super(CNN2, self).__init__()
        self.in_num=in_num
        self.out_num=out_num
        self.conv=nn.Sequential(
            nn.Conv1d(1,8,kernel_size=5,stride=5,padding=0),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv1d(8,32,kernel_size=4,stride=2,padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4,stride=2),

            nn.Conv1d(32,64,kernel_size=4,stride=2,padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4,stride=2)
        )
        self.MLP=nn.Sequential(
            nn.Linear(in_num,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,out_num)
        )
    def forward(self,x):
        x=self.conv(x)
        print("Shape after conv:", x.shape)
        x=x.view(-1,self.in_num)
        print("Shape after conv.view:", x.shape)
        x=self.MLP
        print('shape after mlp:',x.shape)
        x=x.view(-1,self.out_num)
        print('shape after mlp.view:', x.shape)
        return x

if __name__ == '__main__' :
    # create a complete CNN
    model_scratch = CNN2(4090,5)
    print(model_scratch)

# # move tensors to GPU if CUDA is available
# if use_cuda:
#     model_scratch.cuda()