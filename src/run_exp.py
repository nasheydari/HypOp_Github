from src.data_reading import read_uf, read_stanford, read_hypergraph,  read_NDC, read_arxiv
from src.solver import centralized_solver, centralized_solver_for
import logging
import os
import h5py
import numpy as np
import timeit

def exp_centralized(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg":
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                elif params['data'] == "NDC":
                    constraints, header = read_NDC(path)
                else:
                    log.warning('Data mode does not exist. Add the data mode. Current version only support uf, stanford, random_reg, hypergraph, and NDC.')

                res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver(constraints, header, params, file_name)


                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))
                f.create_dataset(f"{file_name}", data = res)



def exp_centralized_for(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')

    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            elif params['data'] == "arxiv":
                constraints, header = read_arxiv()
            else:
                log.warning('Data mode does not exist. Add the data mode. Current version only support uf, stanford, random_reg, hypergraph, arxiv, and NDC.')

            total_nodes = header['num_nodes']
            proc_id = 0
            cur_nodes = list(range(total_nodes * proc_id // 4, total_nodes * (proc_id + 1) // 4))
            cur_nodes = [c+1 for c in cur_nodes]
            inner_constraint = []
            outer_constraint = []
            for c in constraints:
                if c[0] in cur_nodes and c[1] in cur_nodes:
                    inner_constraint.append(c)
                elif (c[0] in cur_nodes and c[1] not in cur_nodes) or (c[0] not in cur_nodes and c[1] in cur_nodes):
                    outer_constraint.append(c)
            res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header, params, file_name, proc_id,
                                                        cur_nodes=cur_nodes, inner_constraint=inner_constraint, outer_constraint=outer_constraint)


            time = timeit.default_timer() - temp_time
            log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')


            with h5py.File(params['res_path'], 'w') as f:
                f.create_dataset(f"{file_name}", data = res)


import torch


def exp_centralized_for_multi(proc_id, devices, params):
    print("start to prepare for device")
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(master_ip="127.0.0.1", master_port="12345")
    torch.cuda.set_device(dev_id)
    TORCH_DEVICE = torch.device("cuda:" + str(dev_id))
    print("start to initialize process")
    torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=len(devices), rank=proc_id)
    print("start to train")

    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')

    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            elif params['data'] == "arxiv":
                constraints, header = read_arxiv()
            else:
                log.warning('Data mode does not exist. Add the data mode. Current version only support uf, stanford, random_reg, hypergraph, arxiv, and NDC.')

            # split the nodes into different devices
            total_nodes = header['num_nodes']
            cur_nodes = list(range(total_nodes * proc_id // len(devices), total_nodes * (proc_id + 1) // len(devices)))
            cur_nodes = [c + 1 for c in cur_nodes]
            inner_constraint = []
            outer_constraint = []
            for c in constraints:
                if c[0] in cur_nodes and c[1] in cur_nodes:
                    inner_constraint.append(c)
                elif (c[0] in cur_nodes and c[1] not in cur_nodes) or (c[0] not in cur_nodes and c[1] in cur_nodes):
                    outer_constraint.append(c)

            print("device", dev_id, "start to train")
            res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header,
                                                                                                params, file_name,
                                                                                             outer_constraint=outer_constraint)

            if res is not None:
                time = timeit.default_timer() - temp_time
                log.info(
                    f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))


                with h5py.File(params['res_path'], 'w') as f:
                    f.create_dataset(f"{file_name}", data=res)