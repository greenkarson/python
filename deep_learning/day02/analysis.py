from day02.net import *
from torch.utils.tensorboard import SummaryWriter
import cv2
net = NetV2()
net.load_state_dict(torch.load("./checkpoint/2.t"))
summaryWriter = SummaryWriter("./logs")
layer1_weight = net.sequential[0].weight
layer2_weight = net.sequential[4].weight
layer3_weight = net.sequential[8].weight

summaryWriter.add_histogram("layer1_weight",layer1_weight)
summaryWriter.add_histogram("layer2_weight",layer2_weight)
summaryWriter.add_histogram("layer3_weight",layer3_weight)
cv2.waitKey(0)