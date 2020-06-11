import numpy as np
import torch, os

w = 226
h = 313
w_ = np.random.randint(-w * 0.2, w * 0.2)
print(w_)

side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
print(side_len)
# 测试字符串浮点数是否能够直接转换为整型数组
# a = ['0.25375939849624063','0.10150375939849623','0.016917293233082706','0.10150375939849623']
# b = list(map(eval,a))
# c = list(map(int,a))
# print(c)


# 注释：
# 例子1--max与maxium的区别

a = [-2, -1, 0, 1, 2]
print(np.max(a))  # 接收一个参数，返回最大值
print(np.maximum(0, a))  # 接收两个参数，X与Y逐个比较取其最大值：若比0小返回0，若比0大返回较大值

b = torch.randn(3, 15, 15)
c = b.unsqueeze(0)
print(c.shape)

# torch.Size([1, 1, 1, 1])
cls = torch.randn(4, 1, 1, 1)
off = torch.randn(4, 4, 1, 1)
blv = cls[0][0]
_off = off[0]
p_cls = 0.6
p = torch.gt(cls, p_cls)
idxs = torch.nonzero(p)
print(p.shape, idxs.shape)
print(blv.shape, _off.shape)

print(torch.randn(1, 1))

a = [["000008.jpg", 212, 89, 218, 302], ["000008.jpg", 212, 89, 218, 302]]
b = [["000008.jpg", 279, 198, 343, 205, 298, 251, 275, 282, 334, 284],
     ["000008.jpg", 279, 198, 343, 205, 298, 251, 275, 282, 334, 284]]
for c in zip(a, b):
    print(list(c))
