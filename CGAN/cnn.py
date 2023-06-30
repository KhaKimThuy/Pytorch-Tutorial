import torch
import torch.nn as nn

def hand(w, chanels, out_channels, kernel_size, stride, padding):
    out = int((w-kernel_size+2*padding)/stride + 1)
    return f'Hand: [{out_channels}, {out}, {out}]'

# # torch.nn.Conv2d(in_channels, out_channels, kernel_size,
# #                 stride=1, padding=0, dilation=1, groups=1, bias=True,
# #                 padding_mode='zeros', device=None, dtype=None)
image = torch.randn(3, 32, 32)
# # out_channels = the number of kernels
# conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
# maxpool1 = nn.MaxPool2d(2,2)
# conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
# maxpool2 = nn.MaxPool2d(2,2)
# # fc1 = nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2)
# # fc2 = nn.Conv2d(120, 84)
# # fc3 = nn.Conv2d(84, 10)
# transpose = nn.ConvTranspose2d(
#     16, 3, kernel_size=4, stride=2, padding=1
# )

# x = conv1(image)
# print(f'Conv1(x) = {x.shape}') # torch.Size([6, 28, 28])
# print(hand(32, 3, 6, 5, 1, 0))


# x = maxpool1(x)
# print(f'Maxpooling1(x) = {x.shape}') # torch.Size([6, 14, 14])

# x = conv2(x)
# print(f'Conv2(x) = {x.shape}') # torch.Size([16, 10, 10])

# x = maxpool2(x)
# print(f'Maxpooling2(x) = {x.shape}') # torch.Size([16, 5, 5])

# # x = x.reshape(-1)
# # print(f'Flatten: {x.shape}')

# x = transpose(x)
a = torch.diag(torch.ones(3))
print(a)