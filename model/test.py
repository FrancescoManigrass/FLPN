import torch
import scipy.io as sio
size = (3, 1)

# creare un array casuale di float
x = torch.rand(size)
y = x
#x = torch.tensor([[1], [2], [3]])
#y = torch.tensor([[-1], [-2], [-3]])
dot_product = torch.sum(torch.multiply(x, y), dim=1)
x = torch.sqrt(torch.sum(torch.square(x), dim=1))
y = torch.sqrt(torch.sum(torch.square(y), dim=1))
similarity = dot_product / (x * y)
print(1 / (1 + torch.exp(-10 * similarity)))


matcontent = sio.loadmat("C:\\Users\\lab2O\\Documents\\Francesco Manigrasso\\polito\\ProtoLTN\\CC_ZSL_VERSION\\data\\wgr\\code\\APN-ZSL-master\\data\\SUN\\att_splits.mat")

print("dfdfd")