import torch
from torch import optim

xs = torch.arange(0.01, 1, 0.01)
# print(xs)
ys = 3 * xs + 4 + torch.randn(99)/100
# print(ys)


class Line(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.w = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.w + self.b


if __name__ == '__main__':

    line = Line()
    opt = optim.SGD(line.parameters(), lr=0.01, momentum=0.01)

    for epoch in range(30):

        for _x, _y in zip(xs, ys):
            z = line(_x)
            loss = (z - _y)**2

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss)

        print(line.b, line.w)



