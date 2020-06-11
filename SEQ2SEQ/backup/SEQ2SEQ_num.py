import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import Sampling_train_num

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        x = x.reshape(-1,180,240).permute(0,2,1)
        x = x.reshape(-1,180)
        fc1 = self.fc1(x)
        fc1 = fc1.reshape(-1, 240, 128)
        lstm,(h_n,h_c) = self.lstm(fc1,None)
        out = lstm[:,-1,:]

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(128,10)

    def forward(self,x):
        x = x.reshape(-1,1,128)
        x = x.expand(-1,4,128)
        lstm,(h_n,h_c) = self.lstm(x,None)
        y1 = lstm.reshape(-1,128)
        out = self.out(y1)
        output = out.reshape(-1,4,10)
        return output


class MainNet (nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder


if __name__ == '__main__':
    BATCH = 64
    EPOCH = 100
    save_path = r'params/seq2seq.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)
    if os.path.exists(os.path.join(save_path)):
        net.load_state_dict(torch.load(save_path))
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Params!")

    train_data = Sampling_train_num.Sampling(root="./code")
    train_loader = data.DataLoader(dataset=train_data,
    batch_size=BATCH, shuffle=True, drop_last=True,num_workers=4)

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            batch_x = x.to(device)
            batch_y = y.float().to(device)

            output = net(batch_x)
            loss = loss_func(output,batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                label_y = torch.argmax(y,2).detach().numpy()
                out_y = torch.argmax(output,2).cpu().detach().numpy()

                accuracy = np.sum(
                out_y == label_y,dtype=np.float32)/(BATCH * 4)
                print("epoch:{},i:{},loss:{:.4f},acc:{:.2f}%"
                .format(epoch,i,loss.item(),accuracy * 100))
                print("label_y:",label_y[0])
                print("out_y:",out_y[0])

        torch.save(net.state_dict(), save_path)
