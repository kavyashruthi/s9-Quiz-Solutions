
class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Sequential(
              nn.Conv2d(in_channels=1,out_channels= 16,kernel_size=(3,3),padding=0,bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              #nn.Conv2d(32, 32, 1, padding=0),
              #nn.ReLU(),
              #nn.BatchNorm2d(32),
              nn.Dropout(0.2)
          )
          self.conv2 = nn.Sequential(
              nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=0,bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              #nn.Conv2d(64, 10, 1, stride=1, padding=0),
              #nn.BatchNorm2d(10),
              #nn.MaxPool2d(2, 2),
              nn.Dropout(0.2)
          )
          
          self.conv3 = nn.Sequential(
              nn.Conv2d(in_channels=32,out_channels=10,kernel_size=(1,1), padding=0,bias=False))
              #nn.ReLU(),
              #nn.BatchNorm2d(128),
          self.pool1= nn.MaxPool2d(2, 2)
              
          
          
          self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2)
        )
          self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2)
        )

          self.conv6=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2)
        )
          self.conv7=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2)       )
          
          self.gap=nn.Sequential(
              nn.AvgPool2d(6)
              )
          
          self.conv8=nn.Sequential(
              nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False)
          )
   #      self.dropout=nn.Dropout(0.2)
  
      def forward(self, x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.conv3(x)
          x = self.pool1(x)
          x = self.conv4(x)
          x = self.conv5(x)
          x = self.conv6(x)
          x = self.conv7(x)
          x = self.gap(x)
          x = self.conv8(x)
          x = x.view(-1,10)
          x = F.log_softmax(x, dim=1)
          return x