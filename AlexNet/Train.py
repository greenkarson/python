import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from AlexNet import AlexNet


class Train():
    def __init__(self, root):
        self.summarywriter = SummaryWriter(log_dir="./runs")
        print("Tensorboard summary writer created")

        self.dataset = datasets.ImageFolder(root, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
        print("Dataset created")
        self.Dataloader = DataLoader(self.dataset,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=2,
                                     pin_memory=False,
                                     drop_last=True,
                                     )
        print("Dataloader created")
        self.net = AlexNet()
        print("AlexNet created")
        self.loss_fn = nn.CrossEntropyLoss()
        print("loss_fn created")
        self.opt = optim.SGD(params=self.net.parameters(),lr=0.01, momentum=0.9,weight_decay=0.0005)
        print("optim created")

    def __call__(self, *args, **kwargs):
        total_step = 1
        lr_scheduler = optim.lr_scheduler.StepLR(self.opt, 30, gamma=0.1)
        print("lr_schedeler created")
        for epoch in range(10000):
            lr_scheduler.step()
            for i, (imgs, classes) in enumerate(self.Dataloader):

                output = self.net(imgs)
                loss = self.loss_fn(output, classes)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if total_step % 10 == 0:
                    with torch.no_grad():
                        _, preds = torch.max(output,dim=1)
                        accuracy = torch.sum(preds==classes)

                        print(f"Epoch:{epoch + 1}----Step:{total_step}----Loss:{loss.item()}----Acc:{accuracy.item()}")
                        self.summarywriter.add_scalar("loss", loss.item(), total_step)
                        self.summarywriter.add_scalar("accurary", accuracy.item(), total_step)

                if total_step % 100 == 0:
                    with torch.no_grad():
                        print("*" * 10)
                        for name, parameter in self.net.named_parameters():
                            if parameter.grad is not None:
                                avg_grad = torch.mean(parameter.grad)
                                print(f"{name} - grad_avg:{avg_grad}")
                                self.summarywriter.add_scalar(f"grad_avg/{name}",avg_grad.item(),total_step)
                                self.summarywriter.add_histogram(f"grad/{name}",parameter.cpu().numpy(),total_step)

                            if parameter.data is not None:
                                avg_weight = torch.mean(parameter.data)
                                print(f"{name} - weight_avg:{avg_weight}")
                                self.summarywriter.add_scalar(f"weight_avg/{name}", avg_weight.item(), total_step)
                                self.summarywriter.add_histogram(f"weight/{name}", parameter.cpu().numpy(), total_step)

            total_step += 1


if __name__ == '__main__':
    train = Train("/Users/karson/Downloads/tiny-imagenet-200/")
    train()
