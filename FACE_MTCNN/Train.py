import torch,glob,os
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from Network import PNet, RNet, ONet
from dataset import MtcnnDataset


class Trainer():

    def __init__(self, net_stage, resume=False):

        self.net_stage = net_stage
        self.train_dataset = MtcnnDataset(r'F:\celeba', net_stage=self.net_stage)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1000, shuffle=True, num_workers=2,
                                           drop_last=True)

        if self.net_stage == 'pnet':
            self.net = PNet()
        if self.net_stage == 'rnet':
            self.net = RNet()
        if self.net_stage == 'onet':
            self.net = ONet()

        if torch.cuda.is_available() is True:
            self.net.cuda()

        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

        self.opt = optim.Adam(self.net.parameters())

        self.epoch_num = 1
        self.global_step = 1

        if resume:
            # torch.load("./param/pnet.pt")
            self.load_state_dict()

        self.summarywrite = SummaryWriter("./runs", purge_step=self.epoch_num)

    def __call__(self):
        for epoch in range(10000):
            total_loss = 0
            for i, (img_data, label, box, landmarks) in enumerate(self.train_dataloader):
                if torch.cuda.is_available() is True:
                    img_data = img_data.cuda()
                    gt_label = label.cuda()
                    gt_boxes = box.cuda()
                    gt_landmarks = landmarks.cuda()
                else:
                    gt_label = label
                    gt_boxes = box
                    gt_landmarks = landmarks

                pred_label, pred_offset, pred_landmarks = self.net(img_data)

                pred_label = pred_label.view(-1, 2)
                pred_offset = pred_offset.view(-1, 4)
                pred_landmarks = pred_landmarks.view(-1, 10)

                # print(pred_label.shape, pred_offset.shape, pred_landmarks.shape)
                # print(label.shape, box.shape, landmarks.shape)

                self.opt.zero_grad()
                cls_loss = self.cls_loss(gt_label, pred_label)
                box_loss = self.box_loss(gt_label, gt_boxes, pred_offset)
                landmark_loss = self.landmark_loss(gt_label, gt_landmarks, pred_landmarks)
                loss = cls_loss + box_loss + landmark_loss
                loss.backward()
                self.opt.step()
                total_loss += loss.cpu().detach()
                # self.summarywrite.add_scalars('train/loss', loss.cpu().detach().item(), global_step=self.epoch_num)
                self.summarywrite.add_scalars("train/loss", {i: j for i, j in
                                                             zip(["loss", "cls_loss", "box_loss", "landmark_loss"],
                                                                 [loss.cpu().detach().item(), cls_loss.cpu().item(),
                                                                  box_loss.cpu().item(), landmark_loss.cpu().item()])
                                                             }, global_step=self.global_step)
                self.global_step += 1

            print(f"epoch:{self.epoch_num}---loss:{loss.cpu().item()}---cls_loss:{cls_loss.cpu().item()}---box_loss:{box_loss.cpu().item()}---landmark_loss:{landmark_loss.cpu().item()}")
            self.save_state_dict()
            self.export_model(f"./param/{self.net_stage}.pt")

            if self.epoch_num % 10 == 0:
                with torch.no_grad():
                    for name, parmeter in self.net.named_parameters():
                        if parmeter.grad is not None:
                            avg_grad = torch.mean(parmeter.grad)
                            print(f"{name}----grad_avg:{avg_grad}")
                            self.summarywrite.add_scalar(f"grad_avg/{name}",avg_grad.item(), self.epoch_num)
                            self.summarywrite.add_histogram(f"grad/{name}",parmeter.cpu().numpy(),self.epoch_num)
                        if parmeter.data is not None:
                            avg_weight = torch.mean(parmeter.data)
                            print(f"{name}----weight_avg:{avg_weight}")
                            self.summarywrite.add_scalar(f"weight_avg/{name}", avg_weight.item(), self.epoch_num)
                            self.summarywrite.add_histogram(f"weight/{name}", parmeter.cpu().numpy(), self.epoch_num)

            self.epoch_num += 1

    def cls_loss(self, gt_label, pred_label):

        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)

        # Online hard sample mining

        mask = torch.eq(gt_label, 0) | torch.eq(gt_label, 1)
        valid_gt_label = torch.masked_select(gt_label, mask)
        mask = torch.stack([mask] * 2, dim=1)
        valid_pred_label = torch.masked_select(pred_label, mask).reshape(-1, 2)

        # compute log-softmax
        # valid_pred_label = torch.log(valid_pred_label)

        loss = self.loss_cls(valid_pred_label, valid_gt_label)

        pos_mask = torch.eq(valid_gt_label, 1)
        neg_mask = torch.eq(valid_gt_label, 0)

        neg_loss = loss.masked_select(neg_mask)
        pos_loss = loss.masked_select(pos_mask)

        if neg_loss.shape[0] > pos_loss.shape[0]:
            neg_loss, _ = neg_loss.topk(pos_loss.shape[0])
        loss = torch.cat([pos_loss, neg_loss])
        loss = torch.mean(loss)

        return loss

    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        mask = torch.eq(gt_label, 1) | torch.eq(gt_label, 2)
        # broadcast mask
        mask = torch.stack([mask] * 4, dim=1)

        # only valid element can effect the loss
        valid_gt_offset = torch.masked_select(gt_offset, mask).reshape(-1, 4)
        valid_pred_offset = torch.masked_select(
            pred_offset, mask).reshape(-1, 4)
        return self.loss_box(valid_pred_offset, valid_gt_offset)

    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label, 3)
        # broadcast mask
        mask = torch.stack([mask] * 10, dim=1)

        valid_gt_landmark = torch.masked_select(
            gt_landmark, mask).reshape(-1, 10)
        valid_pred_landmark = torch.masked_select(
            pred_landmark, mask).reshape(-1, 10)
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)

    def save_state_dict(self):
        checkpoint_name = "checkpoint_epoch_%d" % self.epoch_num
        file_path = f"param/{self.net_stage}/{checkpoint_name}"
        if not os.path.exists(f"param/{self.net_stage}"):
            os.makedirs(f"param/{self.net_stage}")

        state = {
            'epoch_num': self.epoch_num,
            'global_step': self.global_step,
            'state_dict': self.net.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, file_path)

    def export_model(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_state_dict(self):

        # Get the latest checkpoint in output_folder
        all_checkpoints = glob.glob(os.path.join(f"./param/{self.net_stage}", 'checkpoint_epoch_*'))

        if len(all_checkpoints) > 1:
            epoch_nums = [int(i.split('_')[-1]) for i in all_checkpoints]
            max_index = epoch_nums.index(max(epoch_nums))
            latest_checkpoint = all_checkpoints[max_index]

            state = torch.load(latest_checkpoint)
            self.epoch_num = state['epoch_num'] + 1
            # self.global_step = state['self.global_step'] + 1
            self.net.load_state_dict(state['state_dict'])
            self.opt.load_state_dict(state['optimizer'])


if __name__ == '__main__':
    train = Trainer("pnet", resume=True)
    train()
