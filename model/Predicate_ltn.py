import torch
import numpy as np
from torch import sigmoid, nn, softmax

default_layers  =6

class Predicate_ltn(nn.Module):
    def __init__(self, num_features=85,activation=None):
        super(Predicate_ltn, self).__init__()
        self.output_dim = 1
        self.num_features = num_features
        self.k = default_layers

        # Crea un peso trainabile per questo layer
        mn = self.num_features

        self.up = nn.Parameter(torch.ones((self.k, self.output_dim)), requires_grad=True)

        self.Wp = nn.Parameter(torch.randn(self.k, mn, mn), requires_grad=True)
        self.Vp = nn.Parameter(torch.randn(self.k, mn), requires_grad=True)
        self.bp = nn.Parameter(-1 * torch.ones((1, self.k)), requires_grad=True)


        self.parameters = [self.Wp, self.Vp, self.bp, self.up]

    def forward(self, x):
        # (1,batch,features) -> (batch,features)
        X = torch.squeeze(x)
        XW = torch.matmul(torch.tile(torch.unsqueeze(X, 0), [self.k, 1, 1]), self.Wp)
        XWX = torch.squeeze(torch.matmul(torch.unsqueeze(X, 1), XW.permute( 1,2,0)))
        XV = torch.matmul(X, torch.transpose(self.Vp, dim0=0, dim1=1))
        gX = torch.matmul(torch.tanh(XWX + XV + self.bp), self.up)
        h = sigmoid(gX)#torch.sum(softmax(gX) * y, axis=1)

        return h


class Predicate_ltn_same_class(nn.Module):
    def __init__(self,name, num_features, activation=None):
        super(Predicate_ltn_same_class, self).__init__()
        self.output_dim = 1
        self.num_features = num_features
        self.k = default_layers

        # Crea un peso trainabile per questo layer
        mn = self.num_features

        self.up = nn.Parameter(torch.ones((self.k, self.output_dim)), requires_grad=True)

        self.Wp = nn.Parameter(torch.randn(self.k, mn, mn), requires_grad=True)
        self.Vp = nn.Parameter(torch.randn(self.k, mn), requires_grad=True)
        self.bp = nn.Parameter(-1 * torch.ones((1, self.k)), requires_grad=True)


        #self.parameters = torch.nn.ModuleList([self.Wp, self.Vp, self.bp, self.up])

    def forward(self, x):
        # (1,batch,features) -> (batch,features)
        X = torch.squeeze(x)
        XW = torch.matmul(torch.tile(torch.unsqueeze(X, 0), [self.k, 1, 1]), self.Wp)
        XWX = torch.squeeze(torch.matmul(torch.unsqueeze(X, 1), XW.permute( 1,2,0)))
        XV = torch.matmul(X, torch.transpose(self.Vp, dim0=0, dim1=1))
        gX = torch.matmul(torch.tanh(XWX + XV + self.bp), self.up)
        h = sigmoid(gX)#torch.sum(softmax(gX) * y, axis=1)

        return h



class isOfClassLTN(nn.Module):
    def __init__(self, name, num_layers, num_features):
        super(isOfClassLTN, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.k = default_layers

        self.layers =  [Predicate_ltn(num_features=num_features, name="is_of_class_"+i.__str__() , activation=None).cuda() for i in range(self.num_layers)]
        #self.seq = torch.nn.ModuleList(layers).cuda()


    def forward(self, x):
        return torch.cat([ self.layers[f](x) for f in range(len( self.layers))],1)



class Has_attribute(nn.Module):
    def __init__(self, num_layers=8, num_features=85):
        super(Has_attribute, self).__init__()

        self.num_features = num_features
        self.k = default_layers

        self.layers = nn.ModuleList([
            Predicate_ltn(num_features=num_features,  activation=None).cuda() for i in
            range(85)])
        # self.seq = torch.nn.ModuleList(layers).cuda()

    def forward(self, x):
        #return torch.cat([self.layers[f](x) for f in range(len(self.layers))], 1)
        predictions=[]
        for i in self.layers:
            predictions.append(i(x))

        return torch.cat(predictions,axis=1)#torch.gather(torch.cat(predictions,axis=1),1,index)