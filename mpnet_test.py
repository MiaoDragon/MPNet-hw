from __future__ import print_function
from Model.end2end_model import End2EndMPNet
import Model.model as model
import Model.AE.CAE as CAE_2d
import numpy as np
import argparse
import os
import torch
from plan_general import *
import plan_s2d  # planning function specific to s2d environment (e.g.: collision checker, normalization)
import data_loader_2d
from torch.autograd import Variable
import copy
import os
import random
from utility import *
def main(args):
    # set seed
    print(args.model_path)
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # setup evaluation function and load function
    if args.env_type == 's2d':
        IsInCollision = plan_s2d.IsInCollision
        load_test_dataset = data_loader_2d.load_test_dataset
        normalize = plan_s2d.normalize
        unnormalize = plan_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP

    mpNet = End2EndMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, \
                args.output_size, CAE, MLP)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # load previously trained model if start epoch > 0
    model_path='mpnet_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))


    # load test data
    print('loading...')
    test_data = load_test_dataset(N=args.N, NP=args.NP, s=args.s, sp=args.sp, folder=args.data_path)
    obc, obs, paths, path_lengths = test_data

    normalize_func=lambda x: normalize(x, args.world_size)
    unnormalize_func=lambda x: unnormalize(x, args.world_size)

    # test on dataset
    test_suc_rate = 0.
    DEFAULT_STEP = 0.01
    # for statistics
    time_total = []
    time_env = []
    valid_env = []
    fes_env = []

    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in range(len(paths[0])):
            time0 = time.time()
            time_norm = 0.
            fp = 0 # indicator for feasibility
            print ("plannning: env="+str(args.s+i)+" path="+str(args.sp+j))
            if path_lengths[i][j]<2:
                # the data might have paths of length smaller than 2, which are invalid
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>=2:
                fp = 0
                valid_path.append(1)
                path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                        torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]
                step_sz = DEFAULT_STEP
                MAX_NEURAL_REPLAN = 11
                for t in range(MAX_NEURAL_REPLAN):
                    path = neural_plan(mpNet, path, obc[i], obs[i], IsInCollision, \
                                        normalize_func, unnormalize_func, t==0, step_sz=step_sz)
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    if feasibility_check(path, obc[i], IsInCollision, step_sz=0.01):
                        fp = 1
                        print('feasible, ok!')
                        break
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time_path.append(time1)
                print('test time: %f' % (time1))
            # write the path
            if type(path[0]) is not np.ndarray:
                # it is torch tensor, convert to numpy
                path = [p.numpy() for p in path]
            path = np.array(path)
            path_file = args.result_path+'env_%d/' % (i+args.s)
            if not os.path.exists(path_file):
                # create directory if not exist
                os.makedirs(path_file)
            np.savetxt(path_file + 'path_%d.txt' % (j+args.sp), path, fmt='%f')
            fes_path.append(fp)
            print('env %d accuracy up to now: %f' % (i+args.s, (float(np.sum(fes_path))/ np.sum(valid_path))))
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (float(np.sum(fes_env)) / np.sum(valid_env)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',help='folder of trained model')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--NP', type=int, default=200)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--sp', type=int, default=4000)

    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
    parser.add_argument('--AE_input_size', nargs='+', type=int, default=2800, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--result_path', type=str, default='./results/',help='path for saving trained models')
    parser.add_argument('--start_epoch', type=int, default=50)
    parser.add_argument('--env_type', type=str, default='s2d', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')

    args = parser.parse_args()
    print(args)
    main(args)
