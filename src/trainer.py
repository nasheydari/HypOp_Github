from src.model import single_node, single_node_xavier
import timeit
from itertools import chain
import torch
from src.timer import Timer
from src.loss import loss_maxcut_weighted, loss_sat_weighted, loss_maxind_weighted, loss_maxind_QUBO, loss_maxind_weighted2, loss_task_weighted, loss_maxcut_weighted_anealed, loss_task_weighted_vec, loss_maxcut_weighted_multi
from src.utils import  mapping_distribution, gen_q_mis,gen_q_maxcut, mapping_distribution_QUBO, get_normalized_G_from_con, mapping_distribution_vec_task
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import pickle
import time

def centralized_train(X, G, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    elif params['mode'] == 'QUBO_maxcut':
        q_torch = gen_q_maxcut(C, n, torch_dtype=None, torch_device=None)

    temper0=0.01
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}
    X = torch.cat([X[i] for i in X])
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        name=params["model_load_path"]+'conv1_'+file_name[:-4]+'.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"]+'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        conv1 = single_node_xavier(f, f // 2)
        conv2 = single_node_xavier(f // 2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])

    dist=[]
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        dis1=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        dis2=max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.relu(temp)
        temp = conv2(temp)
        dis3 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = G @ temp
        dis4 = max(np.linalg.norm(temp.detach().numpy(), axis=1)) - min(np.linalg.norm(temp.detach().numpy(), axis=1))
        temp = torch.sigmoid(temp)
        dist.append([dis1,dis2,dis3,dis4])
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
        elif params['mode'] == 'maxind':
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            loss = loss_maxind_QUBO(temp, q_torch)

        elif params['mode'] == 'QUBO_maxcut':
            loss = loss_maxind_QUBO(temp, q_torch)
        elif params['mode'] == 'maxcut_annea':
            temper=temper0/(1+i)
            loss = loss_maxcut_weighted_anealed(temp, C, dct, [1 for i in range(len(C))], temper, params['hyper'])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i==int(params['epoch'])-1:
                name=params["model_save_path"]+'embed_'+file_name[:-4]+'.pt'
                torch.save(embed, name)
                name = params["model_save_path"]+'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"]+'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss=loss
    with open("/Users/nasimeh/Documents/distributed_GCN-main-6/Oct12_2023/res/oversmoothing/dist_"+file_name[:-4]+".pkl", "wb") as fp:
        pickle.dump(dist, fp)


    if   params["load best out"]:
        with open("best_out.txt", "r") as f:
            best_out = eval(f.read())
    else:
        best_out = best_out.detach().numpy()
        best_out = {i + 1: best_out[i][0] for i in range(len(best_out))}

    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    if params["mapping"]=="distribution":
        res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    elif params["mapping"]=="threshold":
        res = {x: 0 if best_out[x] < 0.5 else 1 for x in best_out.keys()}

    map_time=timeit.default_timer()-temp_time2
    return res, best_out, train_time, map_time


def centralized_train_for(X, params, f, total_C, n, info_input_total, weights, file_name, device=0,
                          inner_constraint=None, outer_constraint=None, cur_nodes=None, inner_info=None,
                          outer_info=None):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cuda:' + str(device))
    TORCH_DTYPE = torch.float32
    verbose = False

    if inner_constraint is not None:
        total_C = total_C
        C = inner_constraint
        info_input_total = info_input_total
        info_input = inner_info

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p = 0
    count = 0
    prev_loss = 100
    patience = params['patience']
    best_loss = float('inf')
    dct = {x + 1: x for x in range(len(weights))}
    X = torch.cat([X[i] for i in X])

    print("[n]", n, "[C]", len(C), "weight", len(weights))
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
        # parameters=conv2.parameters()
    else:
        embed = nn.Embedding(n, f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        # conv1 = single_node(f, f//2)
        conv1 = single_node_xavier(f, f // 2).to(TORCH_DEVICE)
        conv2 = single_node_xavier(f // 2, 1).to(TORCH_DEVICE)
        # conv2 = single_node(f//2, 1)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        conv1 = conv1.to(TORCH_DEVICE)
        conv2 = conv2.to(TORCH_DEVICE)
        embed = embed.to(TORCH_DEVICE)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())

    # set up multi gpu
    if params["multi_gpu"]:
        if TORCH_DEVICE == torch.device("cpu"):
            conv1 = torch.nn.parallel.DistributedDataParallel(conv1, device_ids=None, output_device=None)
            conv2 = torch.nn.parallel.DistributedDataParallel(conv2, device_ids=None, output_device=None)
        else:
            conv1 = torch.nn.parallel.DistributedDataParallel(conv1, device_ids=[TORCH_DEVICE],
                                                              output_device=TORCH_DEVICE)
            conv2 = torch.nn.parallel.DistributedDataParallel(conv2, device_ids=[TORCH_DEVICE],
                                                              output_device=TORCH_DEVICE)

    if params["multi_gpu"]:
        optimizer = torch.optim.Adam(parameters, lr=params['lr'])
        inputs = embed.weight.to(TORCH_DEVICE)
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(info_input)
    else:
        optimizer = torch.optim.Adam(parameters, lr=params['lr'])
        inputs = embed.weight.to(TORCH_DEVICE)
    if params["test_multi_gpu"] and not params["multi_gpu"]:
        # random select n//4 from info
        selected_indx = random.sample(range(1, n + 1), n // 4)
        info = {i + 1: info_input[selected_indx[i]] for i in range(len(selected_indx))}
        con_list_length = len(selected_indx)
    elif params["multi_gpu"]:
        info = info_input
        con_list_range_keys = list(info.keys())
        con_list_range = [i for i in con_list_range_keys if len(info[i]) > 0]
        print("con_list_range", con_list_range)
    else:
        con_list_length = n
        info = info_input

    start = con_list_range_keys[0]  # start = 1, 501, 1001, 1501
    for ep in range(int(params['epoch'])):
        temp = conv1(inputs)
        temp2 = torch.ones(temp.shape).to(TORCH_DEVICE)
        st_start = time.time()
        st = time.time()

        for i in con_list_range:
            cons_list = info[i]
            indices = [cons[1] if cons[0] == i else cons[0] for cons in cons_list]
            indices_tensor = torch.tensor(indices, dtype=torch.long)  # Convert list to long tensor for indexing
            indices_tensor = indices_tensor - start
            idx = i - start
            temp2[idx, :] += torch.sum(temp[indices_tensor, :], dim=0).to(TORCH_DEVICE)
            temp2[idx, :] /= len(info[i])

        temp = temp2
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp2 = torch.ones(temp.shape).to(TORCH_DEVICE)

        for i in con_list_range:
            cons_list = info[i]
            indices = [cons[1] if cons[0] == i else cons[0] for cons in cons_list]
            indices_tensor = torch.tensor(indices, dtype=torch.long)  # Convert list to long tensor for indexing
            indices_tensor = indices_tensor - start
            idx = i - start
            temp2[idx, :] += torch.sum(temp[indices_tensor, :], dim=0).to(TORCH_DEVICE)
            temp2[idx, :] /= len(info[i])
        temp = temp2
        temp = torch.sigmoid(temp)
        et = time.time()
        if verbose:
            print("Prepare data to compute loss: ", et - st)

        st = time.time()
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            if params["multi_gpu"]:
                temp_reduce = [torch.zeros_like(temp).to(f'cuda:{device}') for _ in range(4)]
                torch.distributed.all_gather(temp_reduce, temp)
                temp_reduce = torch.cat(temp_reduce, dim=0)
                temp_reduce = temp_reduce.squeeze(1)
            loss = loss_maxcut_weighted_multi(temp, C, dct, torch.ones(len(C) + len(outer_constraint)).to(TORCH_DEVICE),
                                        params['hyper'],
                                        TORCH_DEVICE, outer_constraint, temp_reduce, start=start)
        elif params['mode'] == 'maxind':
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        et = time.time()
        if verbose:
            print("Compute forward loss for maxcut: ", et - st)

        st = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        et = time.time()
        if verbose:
            print("Backward loss: ", et - st)

        st = time.time()
        if params["multi_gpu"]:
            loss_sum = loss.clone()
            torch.distributed.reduce(loss_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
            if torch.distributed.get_rank() == 0:
                average_loss = loss_sum / torch.distributed.get_world_size()
                if average_loss < best_loss:
                    print("average_loss", average_loss, "best_loss", best_loss)
                    best_loss = average_loss
                    best_out = temp_reduce.cpu()
                    # print("best_out", best_out)
                    if i == int(params['epoch']) - 1:
                        name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                        torch.save(embed, name)
                        name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                        torch.save(conv1, name)
                        name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                        torch.save(conv2, name)
        else:
            if loss < best_loss:
                best_loss = loss
                best_out = temp
                print(f'found better loss')
                if i == int(params['epoch']) - 1:
                    name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                    torch.save(embed, name)
                    name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                    torch.save(conv1, name)
                    name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                    torch.save(conv2, name)
        prev_loss = loss
        et = time.time()
        if verbose:
            print("Update best loss: ", et - st)
        print("Epoch", ep, "Epoch time: ", et - st_start, "current loss", loss)

    if params["multi_gpu"]:
        if torch.distributed.get_rank() == 0:
            best_out = best_out.detach().numpy()
            print("best_out", best_out)
            best_out = {i + 1: best_out[i] for i in range(len(best_out))}
            with open("best_out.txt", "w") as f:
                f.write(str(best_out))
            train_time = timeit.default_timer() - temp_time
            temp_time2 = timeit.default_timer()
            all_weights = [1.0 for c in (total_C)]
            print("info_input_total", len(info_input_total), "weights", len(weights), "total_C", len(total_C))
            res = mapping_distribution(best_out, params, len(weights), info_input_total, weights, total_C, all_weights,
                                       1, params['penalty'], params['hyper'])
            print("res", res)
            map_time = timeit.default_timer() - temp_time2
        else:
            res = None
            train_time = None
            map_time = None
            best_out = None
    else:
        best_out = best_out.detach().numpy()
        best_out = {i + 1: best_out[i][0] for i in range(len(best_out))}
        train_time = timeit.default_timer() - temp_time
        temp_time2 = timeit.default_timer()
        all_weights = [1.0 for c in (C)]
        res = mapping_distribution(best_out, params, n, info_input, weights, C, all_weights, 1, params['penalty'],
                                   params['hyper'])
        print("res", res)
        map_time = timeit.default_timer() - temp_time2
    return res, best_out, train_time, map_time


def GD_train(X, G, params, f, C, n, info, weights, file_name):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32

    if params['mode'] == 'QUBO':
        q_torch = gen_q_mis(C, n, 2, torch_dtype=None, torch_device=None)
    p=0
    count=0
    prev_loss = 100
    patience=params['patience']
    best_loss = float('inf')
    dct = {x+1: x for x in range(n)}
    X = torch.cat([X[i] for i in X])

    embed = nn.Embedding(n, 1)

    parameters = embed.parameters()
    optimizer = torch.optim.Adam(parameters, lr = params['lr'])
    for i in range(int(params['epoch'])):
        print(i)
        inputs = embed.weight
        temp = torch.sigmoid(inputs)
        if params['mode'] == 'sat':
            loss = loss_sat_weighted(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'maxcut':
            loss = loss_maxcut_weighted(temp, C, dct, [1 for i in range(len(C))], params['hyper'])
        elif params['mode'] == 'maxind':
            #loss = loss_maxind_weighted(temp, C, dct, [1 for i in range(len(C))])
            loss = loss_maxind_weighted2(temp, C, dct, [1 for i in range(len(C))])
        elif params['mode'] == 'QUBO':
            loss = loss_maxind_QUBO(temp, q_torch)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                break
        prev_loss=loss

    best_out = best_out.detach().numpy()
    best_out = {i+1: best_out[i][0] for i in range(len(best_out))}
    train_time = timeit.default_timer()-temp_time
    temp_time2=timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    res = mapping_distribution(best_out, params, n, info, weights, C, all_weights, 1, params['penalty'],params['hyper'])
    map_time=timeit.default_timer()-temp_time2

    return res, best_out, train_time, map_time


def centralized_train_vec_task(X, G, params, f, C, n, info, weights, file_name, C_dic, lenc, leninfo):
    temp_time = timeit.default_timer()
    # fix seed to ensure consistent results
    seed_value = 100
    random.seed(seed_value)  # seed python RNG
    np.random.seed(seed_value)  # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG
    TORCH_DEVICE = torch.device('cpu')
    TORCH_DTYPE = torch.float32


    L=len(C)
    #f = n // 4
    f = 10


    p = 0
    count = 0
    prev_loss = 100
    patience = params['patience']
    best_loss = float('inf')
    dct = {x + 1: x for x in range(n)}
    X = torch.cat([X[i] for i in X])
    if params['transfer']:
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        for param in conv1.parameters():
            param.requires_grad = False
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        for param in conv2.parameters():
            param.requires_grad = False
        parameters = embed.parameters()
    else:
        embed = nn.Embedding(n, L*f)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        conv1 = single_node_xavier(L*f, L*f // 2)
        conv2 = single_node_xavier(L*f // 2, L)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    if params["initial_transfer"]:
        name = params["model_load_path"] + 'conv1_' + file_name[:-4] + '.pt'
        conv1 = torch.load(name)
        name = params["model_load_path"] + 'conv2_' + file_name[:-4] + '.pt'
        conv2 = torch.load(name)
        name = params["model_load_path"] + 'embed_' + file_name[:-4] + '.pt'
        embed = torch.load(name)
        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(parameters, lr=params['lr'])
    for i in range(int(params['epoch'])):
        inputs = embed.weight
        print(i)
        temp = conv1(inputs)
        temp = G @ temp
        temp = torch.relu(temp)
        temp = conv2(temp)
        temp = G @ temp
        temp = torch.sigmoid(temp)
        if params['mode'] == 'task_vec':
            loss = loss_task_weighted_vec(temp, lenc, leninfo)
            if loss == 0:
                print("found zero loss")
                break
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
            count += 1
            if count >= params['patience']:
                print(f'Stopping early on epoch {i} (patience: {patience})')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        else:
            count = 0
        if loss < best_loss:
            p = 0
            best_loss = loss
            best_out = temp
            print(f'found better loss')
            if i == int(params['epoch']) - 1:
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
        else:
            p += 1
            if p > params['patience']:
                print('Early Stopping')
                name = params["model_save_path"] + 'embed_' + file_name[:-4] + '.pt'
                torch.save(embed, name)
                name = params["model_save_path"] + 'conv1_' + file_name[:-4] + '.pt'
                torch.save(conv1, name)
                name = params["model_save_path"] + 'conv2_' + file_name[:-4] + '.pt'
                torch.save(conv2, name)
                break
        prev_loss = loss

    best_out = best_out.detach().numpy()
    best_out_d = {i + 1: best_out[i,:] for i in range(len(best_out))}
    train_time = timeit.default_timer() - temp_time
    temp_time2 = timeit.default_timer()
    all_weights = [1.0 for c in (C)]
    leninfon=torch.Tensor.numpy(leninfo)
    lencn=torch.Tensor.numpy(lenc)
    best_res = mapping_distribution_vec_task(best_out_d, params, n, info, weights, C, C_dic, all_weights, 1, lencn,leninfon,params['penalty'],
                               params['hyper'])
    map_time = timeit.default_timer() - temp_time2
    return best_res, best_out, train_time, map_time


