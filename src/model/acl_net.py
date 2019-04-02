import torch.nn as nn
class ACLNet(nn.Module):
    def __init__(self,c1 = 8,s1 = 5,s2 = 4):
        super().__init__()
        #custom components 1D
        self.conv1 = nn.Conv1d(1,c1,9,s1,padding=4)
        self.conv2 = nn.Conv1d(c1,64,5,s2,padding=2)
        self.maxpool1 = nn.MaxPool1d(int(3200 / (s1 * s2))) 
        self.bn_1d_c1 = nn.BatchNorm1d(c1)
        self.bn_1d_64 = nn.BatchNorm1d(64)

        #custom componentts 2D
        self.conv3 = nn.Conv2d(1,32,3,padding=1)
        self.conv4 = nn.Conv2d(32,64,3,padding=1)
        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.conv6 = nn.Conv2d(64,128,3,padding=1)
        self.conv7 = nn.Conv2d(128,128,3,padding=1)
        self.conv8 = nn.Conv2d(128,256,3,padding=1)
        self.conv9 = nn.Conv2d(256,256,3,padding=1)
        self.conv10 = nn.Conv2d(256,512,3,padding=1)
        self.conv11 = nn.Conv2d(512,512,3,padding=1)
        self.conv12 = nn.Conv2d(512,1,1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn_2d_32 = nn.BatchNorm2d(32)
        self.bn_2d_64 = nn.BatchNorm2d(64)
        self.bn_2d_128 = nn.BatchNorm2d(128)
        self.bn_2d_256 = nn.BatchNorm2d(256)
        self.bn_2d_512 = nn.BatchNorm2d(512)
        self.avgpool = nn.AvgPool2d((2,4))

        #low level and high level features
        self.lff = nn.Sequential(self.conv1,self.bn_1d_c1,nn.ReLU(),self.conv2,self.bn_1d_64,nn.ReLU(),self.maxpool1)
        self.hlf = nn.Sequential(self.conv3,self.bn_2d_32,nn.ReLU(),self.maxpool2,self.conv4,self.bn_2d_64,nn.ReLU(),self.conv5,self.bn_2d_64,nn.ReLU(),
                                 self.maxpool2,self.conv6,self.bn_2d_128,nn.ReLU(),self.conv7,self.bn_2d_128,nn.ReLU(),self.maxpool2,
                                 self.conv8,self.bn_2d_256,nn.ReLU(),self.conv9,self.bn_2d_256,nn.ReLU(),self.maxpool2,self.conv10,self.bn_2d_512,nn.ReLU(),self.conv11,self.bn_2d_512,nn.ReLU(),
                                self.maxpool2,self.conv12,self.avgpool,nn.Sigmoid())
    def forward(self,x):
        out = self.lff(x)
        out = out[:,None,:]
        #out = out.permute(2,1,0)
        #print(out.size())
        out = self.hlf(out)
        return out  
