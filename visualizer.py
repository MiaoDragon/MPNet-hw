import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct
import numpy as np
import argparse

def main(args):
    # visualize point cloud (obstacles)
    obs = []
    for i in range(s,s+N):
    	temp=np.fromfile('data/obs_cloud/obc'+str(i)+'.dat')
    	obs.append(temp)
    obs = np.array(obs).astype(np.float32)

    plt.scatter(obc[:,0], obc[:,1])




    # visualize path
    path = np.fromfile(args.path_file)
    path = path.reshape(len(path)//2, 2)
    path_x = []
    path_y = []
    for i in range(len(path)):
        path_x.append(path[i][0])
        path_y.append(path[i][1])
    plt.plot(path_x, path_y)

    plt.show()


parser = argparse.ArgumentParser()
# for training
parser.add_argument('--path_file', type=str, default='./data/path_0.txt',help='path for saving trained models')
args = parser.parse_args()
print(args)
main(args)
