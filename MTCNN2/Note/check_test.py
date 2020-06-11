import numpy as np
strs = ['000001.jpg','95','71','226','313']
x1 = int(strs[1])
y1 = int(strs[2])
w = int(strs[3])
h = int(strs[4])
x2 = x1 + w
y2 = y1 + h


x1 = int(x1 + w * 0.12)
y1 = int(y1 + h * 0.1)
x2 = int(x1 + w * 0.9)
y2 = int(y1 + h * 0.85)
w = x2 - x1
h = y2 - y1
print(x1,y1,x2,y2)
cx = int(x1 + (w / 2))
cy = int(y1 + (w / 2))

_cx = cx + np.random.randint(-w * 0.2, w * 0.2)
_cy = cy + np.random.randint(-h * 0.2, h * 0.2)
_w = w + np.random.randint(-w * 0.2, w * 0.2)
_h = h + np.random.randint(-h * 0.2, h * 0.2)
_x1 = int(_cx - (_w / 2))
_y1 = int(_cy - (_h / 2))
_x2 = int(_x1 + _w)
_y2 = int(_y1 + _h)
print(_x1,_y1,_x2,_y2)
_x1_off = (_x1 - x1) / _w
_y1_off = (_y1 - y1) / _h
_x2_off = (_x2 - x2) / _w
_y2_off = (_y2 - y2) / _h
print(_x1_off,_y1_off,_x2_off,_y2_off)
offset = [_x1_off,_y1_off,_x2_off,_y2_off]
x1 = (_x1 - offset[0] * _w)
y1 = (_y1 - offset[1] * _h)
x2 = (_x2 - offset[2] * _w)
y2 = (_y2 - offset[3] * _h)
print(x1,y1,x2,y2)