import torch,os

print(torch.cuda.is_available())
right = torch.tensor(226).cuda()
total = 246
tp = torch.tensor(62).cuda()
fp = torch.tensor(10).cuda()
tn = torch.tensor(64).cuda()
fn = torch.tensor(10).cuda()
print(right, total, tp, fp, tn, fn)
# acc = right / total
acc = torch.true_divide(right,total)
print(acc)