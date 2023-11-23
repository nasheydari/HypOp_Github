from src.utils import generate_H_from_edges, _generate_G_from_H,  all_to_weights, all_to_weights_task, gen_q_mis, get_normalized_G_from_con, Maxind_postprocessing, sparsify_graph
import numpy as np
import torch
from src.sampler import Sampler
import timeit
from src.trainer import centralized_train, GD_train, centralized_train_for, centralized_train_vec_task
from src.loss import loss_maxcut_numpy_boost, loss_sat_numpy_boost, loss_maxind_numpy_boost, loss_maxind_QUBO, loss_task_numpy_vec
import numpy as np
import torch




def centralized_solver(constraints, header, params, file_name):
    temp_time = timeit.default_timer()
    n = header['num_nodes']
    if params['data'] != 'hypergraph' and  params['data'] != 'task' and params['data'] != 'uf' and params['data'] != 'NDC':
        q_torch = gen_q_mis(constraints, n, 2, torch_dtype=None, torch_device=None)
    f = int(np.sqrt(n))
    #f=n // 2


    info = {x + 1: [] for x in range(n)}
    for constraint in constraints:
        if params['data'] == 'task':
            for node in constraint[:-1]:
                info[abs(node)].append(constraint)
        else:
            for node in constraint:
                info[abs(node)].append(constraint)

    sparsify = params['sparsify']
    spars_p=params['sparsify_p']
    if sparsify:
        constraints_sparse, header_sparse, info_sparse = sparsify_graph(constraints, header, info, spars_p)
    else:
        constraints_sparse, header_sparse, info_sparse = constraints, header, info

    if params['data']!='task':
        edges = [[abs(x) - 1 for x in edge] for edge in constraints_sparse]
    else:
        edges = [[abs(x) - 1 for x in edge[:-1]] for edge in constraints_sparse]


    load_G= params['load_G']
    if params['random_init']=='none' and not load_G:
        if params['data'] != 'hypergraph' and params['data'] != 'task' and params['data'] != 'uf' and params['data'] != 'NDC':
            G = get_normalized_G_from_con(constraints_sparse, header_sparse)
        else:
            H = generate_H_from_edges(edges, n)
            G = _generate_G_from_H(H)
            name_g="./models/G/"+params['mode']+'_'+file_name[:-4]+".npy"
            with open(name_g, 'wb') as ffff:
                np.save(ffff,G)
            G = torch.from_numpy(G).float()
    elif load_G:
        name_g="./models/G/"+params['mode']+'_'+file_name[:-4]+".npy"
        G=np.load(name_g)
        G = torch.from_numpy(G).float()
    else:
        G=torch.zeros([n,n])


    all_weights = [[1.0 for c in (constraints)] for i in range(params['num_samples'])]
    if params['data'] != 'task':
        weights = [all_to_weights(all_weights[i], n, constraints) for i in range(len(all_weights))]
    else:
        weights = [all_to_weights_task(all_weights[i], n, constraints) for i in range(len(all_weights))]
    # sampler initialization
    sampler = Sampler(params, n, f)
    # timer initialization            
    reses = []
    reses2 = []
    reses_th = []
    probs = []
    for i in range(params['K']):
        #print(weights)
        scores = []
        scores2 = []
        scores_th = []
        scores1 = []
        Xs = sampler.get_Xs(i)
        temp_weights = []
        for j in range(params['num_samples']):
            #res, res2, prob = centralized_train(Xs[j], Gn, params, f, constraints, n, info, weights[i])
            if params["mode"]=='task_vec':
                C_dic = {}
                ic = 0
                lenc=torch.zeros([len(constraints)])
                for c in constraints:
                    lenc[ic]=len(c)
                    C_dic[str(c)] = ic
                    ic += 1

                leninfo=torch.zeros([n])
                for inn in range(n):
                    leninfo[inn]=len(info[inn+1])
                
                res, prob, train_time, map_time = centralized_train_vec_task(Xs[j], G, params, f, constraints, n, info, weights[i], file_name, C_dic, lenc, leninfo)
            elif not params["GD"]:
                res,  prob , train_time, map_time= centralized_train(Xs[j], G, params, f, constraints, n, info, weights[i], file_name)
            else:
                res, prob, train_time, map_time = GD_train(Xs[j], G, params, f, constraints, n, info, weights[i], file_name)
            if params["mode"]!='task_vec':
                res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}
            else:
                res_th = {x: [0 if prob[x,i]<0.5 else 1 for i in range(len(constraints))] for x in range(n)}

            if params['mode'] == 'sat':
                score, new_w = loss_sat_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                scores.append(score)
                score, new_w = loss_sat_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],                                 inc=params['boosting_mapping'])
                scores_th.append(score)
            elif params['mode'] == 'maxcut' or params['mode'] == 'QUBO_maxcut' or params['mode'] == 'maxcut_annea':
                score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                score_th, _ =  loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                scores.append(score)
                scores_th.append(score_th)
            elif params['mode'] == 'maxind':
                res_feas=Maxind_postprocessing(res,constraints, n)
                score, score1, new_w = loss_maxind_numpy_boost(res_feas, constraints, [1 for i in range(len(constraints))], inc=params['boosting_mapping'])
                score_th, score1, new_w = loss_maxind_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],inc=params['boosting_mapping'])
                print(score, score1)
                scores.append(score)
                scores1.append(score1)
                scores_th.append(score_th)
            elif params['mode'] == 'QUBO':
                res_feas = Maxind_postprocessing(res, constraints, n)
                res_th_feas = Maxind_postprocessing(res_th, constraints, n)
                score = loss_maxind_QUBO(torch.Tensor(list(res_feas.values())), q_torch)
                score_th = loss_maxind_QUBO(torch.Tensor(list(res_th_feas.values())), q_torch)
                scores.append(score)
                scores_th.append(score_th)
            elif params['mode'] == 'task_vec':
                leninfon = torch.Tensor.numpy(leninfo)
                lencn = torch.Tensor.numpy(lenc)
                score = loss_task_numpy_vec(res, lencn, leninfon)
                res_th_array = np.array(list(res_th.values()))
                score_th = loss_task_numpy_vec(res_th_array, lencn, leninfon)
                scores.append(score)
                scores_th.append(score_th)
            probs.append(prob)
        sampler.update(scores)
        reses.append(scores)
        reses_th.append(scores_th)
    return reses, reses2, reses_th, probs, timeit.default_timer() - temp_time, train_time, map_time


def centralized_solver_for(constraints, header, params, file_name, device=0,
                           cur_nodes=None, inner_constraint=None, outer_constraint=None):
    temp_time = timeit.default_timer()
    edges = [[abs(x) - 1 for x in edge] for edge in constraints]
    if cur_nodes is None:
        n = header['num_nodes']
    else:
        n = len(cur_nodes)

    f = int(np.sqrt(n))
    # f=n // 2

    info = {x + 1: [] for x in range(header['num_nodes'])}
    inner_info = {x: [] for x in cur_nodes}
    outer_info = {x: [] for x in cur_nodes}

    if cur_nodes is None:
        for constraint in constraints:
            for node in constraint:
                info[abs(node)].append(constraint)
    else:
        for constraint in inner_constraint:
            for node in constraint:
                inner_info[abs(node)].append(constraint)
        for constraint in outer_constraint:
            for node in constraint:
                if node in cur_nodes:
                    outer_info[abs(node)].append(constraint)

        for constraint in constraints:
            for node in constraint:
                info[abs(node)].append(constraint)
    all_weights = [[1.0 for c in (inner_constraint)] for i in range(params['num_samples'])]

    weights = [all_to_weights(all_weights[i], header['num_nodes'], inner_constraint) for i in range(len(all_weights))]
    # sampler initialization
    sampler = Sampler(params, n, f)
    # timer initialization
    reses = []
    reses2 = []
    reses_th = []
    probs = []
    for i in range(params['K']):
        scores = []
        scores2 = []
        scores_th = []
        scores1 = []
        Xs = sampler.get_Xs(i)
        temp_weights = []
        for j in range(params['num_samples']):
            res, prob, train_time, map_time = centralized_train_for(Xs[j], params, f, constraints, n, info,
                                                                        weights[i], file_name, device
                                                                        , inner_constraint, outer_constraint, cur_nodes,
                                                                        inner_info, outer_info)

            if torch.distributed.get_rank() == 0:
                res_th = {x: 0 if prob[x] < 0.5 else 1 for x in prob.keys()}
                if params['mode'] == 'sat':
                    score, new_w = loss_sat_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                        inc=params['boosting_mapping'])
                    scores.append(score)
                elif params['mode'] == 'maxcut':
                    score, new_w = loss_maxcut_numpy_boost(res, constraints, [1 for i in range(len(constraints))],
                                                           inc=params['boosting_mapping'])
                    score_th, _ = loss_maxcut_numpy_boost(res_th, constraints, [1 for i in range(len(constraints))],
                                                          inc=params['boosting_mapping'])
                    scores.append(score)
                    scores_th.append(score_th)
                elif params['mode'] == 'maxind':
                    res_feas = Maxind_postprocessing(res, constraints, n)
                    score, score1, new_w = loss_maxind_numpy_boost(res, constraints,
                                                                   [1 for i in range(len(constraints))],
                                                                   inc=params['boosting_mapping'])
                    score_th, score1, new_w = loss_maxind_numpy_boost(res_th, constraints,
                                                                      [1 for i in range(len(constraints))],
                                                                      inc=params['boosting_mapping'])
                    print(score, score1)
                    scores.append(score)
                    scores1.append(score1)
                    scores_th.append(score_th)


            if torch.distributed.get_rank() == 0:
                probs.append(prob)
        if torch.distributed.get_rank() == 0:
            sampler.update(scores)
            reses.append(scores)
            reses_th.append(scores_th)
    if torch.distributed.get_rank() == 0:
        return reses, reses2, reses_th, probs, timeit.default_timer() - temp_time, train_time, map_time
    else:
        return None, None, None, None, None, None, None




