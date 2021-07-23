import torch

class FFN(torch.nn.Module):
    """ Some class description. """
    def __init__(self):
        super(FFN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(in_features=8*8*128, out_features=256, bias=True)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=64, bias=True)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=10, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)
        #self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  # 32x#32#64
        x = torch.nn.functional.relu(self.conv2(x))  # 32x32x128
        x = self.pool(x)    # 16x16x128
        x = torch.nn.functional.relu(self.conv3(x))  # 16*16*256
        x = torch.nn.functional.relu(self.conv4(x))  # 16*16*256
        x = self.pool(x)    #8x8x256
        batch_size = x.shape[0]
        x = x.reshape([batch_size, -1])
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        nosoftmax = self.fc3(x)
        return self.softmax(nosoftmax), nosoftmax