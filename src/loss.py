import torch
import torch.nn.functional as F
import timeit
import numpy as np

def loss_maxcut_weighted(probs, C, dct, weights, hyper=False):
    x = probs.squeeze()
    loss = 0
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        temp = (temp_1s + temp_0s)
        loss += (temp * w)
    return loss


def loss_maxcut_weighted_multi(probs, C, dct, weights_tensor, hyper=False, TORCH_DEVICE='cpu', outer_constraint=None,
                         temp_reduce=None, start=0):
    x = probs.squeeze()
    loss = 0
    inner_temp_values = torch.zeros(len(outer_constraint) + len(C)).to(TORCH_DEVICE)
    out_point = len(C)
    total_C = C + outer_constraint
    for idx, c in enumerate(total_C):
        if hyper:
            indices = [dct[index] for index in c]
        else:
            indices = [dct[index] for index in c[0:2]]
        '''
        indices = [index-start for index in indices]
        selected_x = x[indices]
        temp_1s = torch.prod(1 - selected_x)
        temp_0s = torch.prod(selected_x)
        inner_temp_values[idx] =  temp_1s + temp_0s - 1
        '''
        indices = [index - start + 1 for index in indices]
        if idx < out_point:
            selected_x = x[indices]
            temp_1s = torch.prod(1 - selected_x)
            temp_0s = torch.prod(selected_x)
            inner_temp_values[idx] = temp_1s + temp_0s - 1
        else:
            res = [x[indice] if indice >= 0 and indice < len(x) else temp_reduce[indice + start - 1] for indice in
                   indices]
            selected_x = torch.stack(res)
            temp_1s = torch.prod(1 - selected_x)
            temp_0s = torch.prod(selected_x)
            inner_temp_values[idx] = 0.5 * (temp_1s + temp_0s - 1)


    loss = torch.sum(inner_temp_values * weights_tensor)
    return loss

def loss_maxcut_weighted_anealed(probs, C, dct, weights,  temper, hyper=False):
    x = probs.squeeze()
    loss = 0
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[dct[index]])
                temp_0s *= (x[dct[index]])
        temp = (temp_1s + temp_0s)
        loss += (temp * w)
    Entropy=sum([item* torch.log2(item)+(1-item)*torch.log2(1-item) for item in x])
    loss+= temper*Entropy
    return loss


def loss_task_weighted(res, C, dct, weights):
    loss = 0
    x = res.squeeze()
    new_w = []
    for c, w in zip(C, weights):
        temp=sum([x[dct[index]] for index in c[:-1]])
        if c[-1]=='T':
            temp1 = w * max(np.ceil((len(c)-1)/2)-temp , 0)
        else:
            temp1 = w * max(temp - np.ceil((len(c)-1)/2) , 0 )
        loss = loss+temp1
    temp2 = 0.2*sum([min(1 - x[index], x[index]) for index in range(len(x))])
    loss = loss + temp2
    return loss


def loss_maxind_weighted(probs, C, dct, weights):
    p=4
    x = probs.squeeze()
    loss = - sum(x)
    for c, w in zip(C, weights):
        temp = (p * w * x[dct[c[0]]] * x[dct[c[1]]])
        loss += (temp)
    return loss

def loss_maxind_weighted2(probs, C, dct, weights):
    p=4
    x = probs.squeeze()
    loss = - (x.T@x)
    for c, w in zip(C, weights):
        temp = (p * w * x[dct[c[0]]] * x[dct[c[1]]])
        loss += (temp)
    return loss

def loss_sat_weighted(probs, C, dct, weights):
    x = probs.squeeze()
    loss = 0
    for c, w in zip(C, weights):
        temp = 1
        for index in c:
            if index > 0:
                temp *= (1 - x[dct[abs(index)]])
            else:
                temp *= (x[dct[abs(index)]])
        loss += (temp * w)
    return loss



def loss_sat_numpy(res, C, weights, penalty=0, hyper=True):
    loss = 0
    for c, w in zip(C, weights):
        temp = 1
        for index in c:
            if index > 0:
                temp *= (1 - res[abs(index)])
            else:
                temp *= (res[abs(index)])
        loss += (temp * w)
    return loss

def loss_sat_numpy_boost(res, C, weights, inc=1.1):
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp = 1
        for index in c:
            if index > 0:
                temp *= (1 - res[abs(index)])
            else:
                temp *= (res[abs(index)])
        loss += (temp)
        if temp >= 1:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss, new_w

# for mapping, maxcut
def loss_maxcut_numpy(x, C, weights, penalty=0, hyper=False):
    loss = 0
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        if hyper:
            for index in c:
                temp_1s *= (1 - x[index])
                temp_0s *= (x[index])
        else:
            for index in c[0:2]:
                temp_1s *= (1 - x[index])
                temp_0s *= (x[index])
        temp = (temp_1s + temp_0s - 1)
        loss += temp
    if loss>-2:
        loss += penalty
    return loss


# for mapping, maxind
def loss_maxind_numpy(x, C, weights, penalty=0, hyper=False):
    p=4
    loss = - sum(x.values())
    for c, w in zip(C, weights):
        temp = p * w * x[c[0]] * x[c[1]]
        loss += (temp)
    return loss


def loss_maxcut_numpy_boost(res, C, weights, inc=1.1):
    loss = 0
    new_w = []
    for c, w in zip(C, weights):
        temp_1s = 1
        temp_0s = 1
        for index in c:
                temp_1s *= (1 - res[index])
                temp_0s *= (res[index])
        temp = (temp_1s + temp_0s - 1)
        loss += (temp)
        if temp >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss, new_w


def loss_maxind_numpy_boost(res, C, weights, inc=1.1):
    p=4
    new_w = []
    loss1 = - sum(res.values())
    loss = - sum(res.values())
    for c, w in zip(C, weights):
        temp = p * w * res[c[0]] * res[c[1]]
        loss += (temp)
        if temp >= 0:
            new_w.append(w * inc)
        else:
            new_w.append(w)
    return loss, loss1, new_w


def loss_task_weighted_vec(res, lenc, leninfo):
    x_c = res.squeeze()
    [n,m]=x_c.shape
    temp=lenc-torch.sum(x_c,0)
    temp1 = torch.maximum(temp, torch.zeros([m]))
    temp1s=sum(temp1)
    temp2=torch.sum(x_c,1)-leninfo-50*torch.ones([n])
    temp3=torch.maximum(temp2, torch.zeros([n]))
    temp3s=sum(temp3)
    loss = temp1s + temp3s
    return loss

def loss_task_numpy_vec(res, lenc, leninfo):
    x_c=res
    [n,m]=x_c.shape
    temp=lenc-np.sum(x_c,axis=0)
    temp1 = np.maximum(temp, np.zeros([m,]))
    temp1s=sum(temp1)
    temp2=np.sum(x_c,axis=1)-leninfo-50*np.ones([n])
    temp3=np.maximum(temp2, np.zeros([n,]))
    temp3s=sum(temp3)
    loss = temp1s + temp3s
    return loss


def loss_maxind_QUBO(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """
    cost = (probs.T @ Q_mat @ probs).squeeze()

    return cost