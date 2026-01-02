import torch
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from math import pi
from Test.utils import *

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def pareto_efficient_weights(prev_w,c,G):
    K = G.shape[0]
    GGT = np.matmul(G,np.transpose(G))
    e = np.ones(np.shape(prev_w))

    m_up = np.hstack((GGT,e))
    M = np.vstack((m_up,m_down))

    m_up = np.hstack((GGT,e))
    m_down = np.hstack((np.transpose(e),np.zeros((1,1))))
    M = np.vstack((m_up,m_down))

    z = np.vstack((-np.matmul(GGT,c), 1 - np.sum(c)))

    MTM = np.matmul(np.transpose(M),M)
    w_hat = np.matmul(np.matmul(np.linalg.inv(MTM),M),z)
    w_hat = w_hat[:-1]
    w_hat = np.reshape(w_hat,(w_hat.shape[0],))

    return active_set_method(w_hat,prev_w,c)
def active_set_method(w_hat,prev_w,c):
    A = np.eye(len(c))
    cons = {'type':'eq','fun':lambda x: np.sum(x) - 1}
    bounds = [[0,None] for _ in range(len(w_hat))]
    result = minimize(lambda x:np.linalg.norm(A.dot(x) - w_hat),x0=prev_w.flatten(),
                      method='SLSQP',
                      bounds = bounds,
                      constraints = cons)
    return result.x + c
ph_wa = tf.placeholder(tf.float32)
ph_wb = tf.placeholder(tf.float32)

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self,weight=None,reduction='mean',smoothing=0.0):
        super().__init__(weight=weight,reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def multilabel_categorical_crossentropy(y_true,y_pred):
        y_pred = (1 - 2* y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[...,:1])
        y_pred_neg = torch.cat([y_pred_neg,zeros],dim=-1)
        y_pred_pos = torch.cat([y_pred_pos,zeros],dim=-1)
        neg_loss = torch.logsumexp(y_pred_pos,dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos,dim=-1)
        return neg_loss + pos_loss
    
    def _smooth_one_hot(targets:torch.Tensor,n_classes:int,smoothing=0.0):
        assert 0 <=smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0),n_classes),
                                  device = targets.device) \
                .fill_(smoothing / n_classes - 1) \
                .scatter_(1,targets.data.unsqueeze(1),1. - smoothing)
            return  targets            

class CenterLoss(nn.Module):
    def __init__(self,num_classes = 10,feat_dim = 2,use_gpu = True):
        super(CenterLoss,self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes,self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes,self.feat_dim))

    def forward(self,x,labels):
        batch_size = x.size(0)
        distmat = torch.pow(x,2).sum(dim=1,keepdim=True).expand(batch_size,self.num_classes) + \
                  torch.pow(self.centers,2).sum(dim=1,keepdim=True).expand(self.num_classes,batch_size).t()
        distmat.addmm_(1,-2,x,self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size,self.num_classes)
        mask = labels.eq(classes.expand(batch_size,self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12,max=1e+12,).sum() / batch_size

        return loss

loss_a = LabelSmoothCrossEntropyLoss(num_classes=2,eps=0.1,reduction='mean')
loss_b = CenterLoss(num_classes = 2,feat_dim = 256,lambda_c = 0.5)
loss = ph_wa * loss_a + ph_wb * loss_b

a_gradients = tf.gradients(loss_a)
b_gradients = tf.gradients(loss_b)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(loss)


sess = tf.Session()

w_a,w_b = 0.5,0.5
c = c = tf.constant(0.01, dtype=tf.float32)
for step in range(0,320):
    res = sess.run([a_gradients,b_gradients,train_op],feed_dict = {ph_wa:w_a,ph_wa:w_b})

    G = np.hstack(([res[0][0],res[1][0]]))
    G = np.transpose(G)

    w_a,w_b = pareto_efficient_weights(prev_w=np.asarray(w_a,w_b),c = c,G = G)
    





    