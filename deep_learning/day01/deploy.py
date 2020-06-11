from torch import jit
from day01.net import *
if __name__ == '__main__':
    modle = NetV1()
    modle.load_state_dict(torch.load("./checkpoint/4.t"))

    input = torch.rand(1,784)

    traced_script_moudle = torch.jit.trace(modle,input)
    traced_script_moudle.save("mnist.pt")