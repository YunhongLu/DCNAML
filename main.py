from __future__ import print_function

import sys
import os
import time
import timeit
import numpy as np
import theano

theano.config.floatX= 'float32'

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score

try:
    import cPickle as pickle
except:
    import pickle
import h5py
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
try:
    from six.moves import xrange
except:
    pass

import scipy
from numpy.matlib import repmat
from scipy.spatial.distance import cdist
from scipy import sparse

import torchvision
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random

USE_CUDA = torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)


def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):

    num_i = np.size(cluster_i)
    num_j = np.size(cluster_j)

    # detecting cross elements (this check costs much and is unnecessary)

    ijGroupIndex = np.append(cluster_i, cluster_j)

    y_ij = np.zeros((num_i + num_j, 2))  # [y_i, y_j]
    y_ij[:num_i, 0] = 1
    y_ij[num_i:, 1] = 1
    idx = np.ix_(ijGroupIndex, ijGroupIndex)
    L_ij = scipy.linalg.inv(IminuszW[idx]).dot(y_ij)
    L_ij = sum(L_ij[:num_i, 0]) / (num_i * num_i) + sum(L_ij[num_i:, 1]) / (num_j * num_j)

    return L_ij


def gacPathEntropy(subIminuszW):

    N = subIminuszW.shape[0]
    clusterComp = scipy.linalg.inv(subIminuszW).dot(np.ones((N, 1)))
    clusterComp = sum(clusterComp) / (N * N)

    return clusterComp


def gacMerging(graphW, initClusters, groupNumber, strDescr, z):

    numSample = graphW.shape[0]
    IminuszW = np.eye(numSample) - z * graphW
    myInf = 1e10

    # initialization
    VERBOSE = True

    numClusters = len(initClusters)
    if numClusters <= groupNumber:
        print('GAC: too few initial clusters. Do not need merging!');

    # compute the structural complexity of each initial cluster
    clusterComp = np.zeros((numClusters, 1))
    for i in xrange(numClusters):
        clusterComp[i] = gacPathEntropy(IminuszW[np.ix_(initClusters[i], initClusters[i])])

    # compute initial(negative) affinity table(upper trianglar matrix), very slow
    if VERBOSE:
        print('   Computing initial table.')

    affinityTab = np.full(shape=(numClusters, numClusters), fill_value=np.inf)
    for j in xrange(numClusters):
        for i in xrange(j):
            affinityTab[i, j] = -1 * gacPathCondEntropy(IminuszW, initClusters[i], initClusters[j])

    affinityTab = (clusterComp + clusterComp.T) + affinityTab

    if VERBOSE:
        print('   Starting merging process')

    curGroupNum = numClusters
    while True:
        if np.mod(curGroupNum, 20) == 0 & VERBOSE:
            print('   Group count: ', str(curGroupNum))

        # Find two clusters with the best affinity
        minAff = np.min(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex2 = np.argmin(minAff)
        minIndex1 = minIndex1[minIndex2]
        if minIndex2 < minIndex1:
            minIndex1, minIndex2 = minIndex2, minIndex1

        # merge the two clusters

        new_cluster = np.unique(np.append(initClusters[minIndex1], initClusters[minIndex2]))

        # move the second cluster to be merged to the end of the cluster array
        # note that we only need to copy the end cluster's information to
        # the second cluster 's position
        if minIndex2 != curGroupNum:
            initClusters[minIndex2] = initClusters[-1]
            clusterComp[minIndex2] = clusterComp[curGroupNum - 1]
            # affinityTab is an upper triangular matrix
            affinityTab[: minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum - 1]
            affinityTab[minIndex2, minIndex2 + 1: curGroupNum - 1] = affinityTab[minIndex2 + 1:curGroupNum - 1,
                                                                     curGroupNum - 1]

        # update the first cluster and remove the second cluster
        initClusters[minIndex1] = new_cluster
        initClusters.pop()
        clusterComp[minIndex1] = gacPathEntropy(IminuszW[np.ix_(new_cluster, new_cluster)])
        clusterComp[curGroupNum - 1] = myInf
        affinityTab[:, curGroupNum - 1] = myInf
        affinityTab[curGroupNum - 1, :] = myInf
        curGroupNum = curGroupNum - 1
        if curGroupNum <= groupNumber:
            break

        # update the affinity table for the merged cluster
        for groupIndex1 in xrange(minIndex1):
            affinityTab[groupIndex1, minIndex1] = -1 * gacPathCondEntropy(IminuszW, initClusters[groupIndex1],
                                                                          new_cluster)
        for groupIndex1 in xrange(minIndex1 + 1, curGroupNum):
            affinityTab[minIndex1, groupIndex1] = -1 * gacPathCondEntropy(IminuszW, initClusters[groupIndex1],
                                                                          new_cluster)
        affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1].reshape(-1) + clusterComp[minIndex1] + affinityTab[
                                                                                                            :minIndex1,
                                                                                                            minIndex1]
        affinityTab[minIndex1, minIndex1 + 1: curGroupNum] = clusterComp[minIndex1 + 1: curGroupNum].T + clusterComp[
            minIndex1] + affinityTab[minIndex1, minIndex1 + 1:curGroupNum]

    # generate sample labels
    clusterLabels = np.ones((numSample, 1))
    for i in xrange(len(initClusters)):
        clusterLabels[initClusters[i]] = i
    if VERBOSE:
        print('   Final group count: ', str(curGroupNum))

    return clusterLabels


def gacNNMerge(distance_matrix, NNIndex):

    # NN indices
    sampleNum = distance_matrix.shape[0]
    clusterLabels = np.zeros((sampleNum, 1))
    counter = 1
    for i in xrange(sampleNum):
        idx = NNIndex[i, :2]
        assignedCluster = clusterLabels[idx]
        assignedCluster = np.unique(assignedCluster[np.where(assignedCluster > 0)])
        if len(assignedCluster) == 0:
            clusterLabels[idx] = counter
            counter = counter + 1
        elif len(assignedCluster) == 1:
            clusterLabels[idx] = assignedCluster
        else:
            clusterLabels[idx] = assignedCluster[0]
            for j in xrange(1, len(assignedCluster)):
                clusterLabels[np.where(clusterLabels == assignedCluster[j])] = assignedCluster[0]

    uniqueLabels = np.unique(clusterLabels)
    clusterNumber = len(uniqueLabels)

    initialClusters = []
    for i in xrange(clusterNumber):
        initialClusters.append(np.where(clusterLabels[:].flatten() == uniqueLabels[i])[0])

    return initialClusters


def gacBuildDigraph(distance_matrix, K, a):

    N = distance_matrix.shape[0]
    # find 2*K NNs in the sense of given distances
    sortedDist = np.sort(distance_matrix, axis=1)
    NNIndex = np.argsort(distance_matrix, axis=1)
    NNIndex = NNIndex[:, :K + 1]

    # estimate derivation
    sig2 = np.mean(np.mean(sortedDist[:, 1:max(K + 1, 4)])) * a
    #########
    tmpNNDist = np.min(sortedDist[:, 1:], axis=1)
    while any(np.exp(- tmpNNDist / sig2) < 1e-5):  # check sig2 and magnify it if it is too small
        sig2 = 2 * sig2

    #########
    print('  sigma = ', str(np.sqrt(sig2)))

    # build graph
    ND = sortedDist[:, 1:K + 1]
    NI = NNIndex[:, 1:K + 2]
    XI = repmat(np.arange(0, N).reshape(-1, 1), 1, K)
    sig2 = np.double(sig2)
    ND = np.double(ND)
    graphW = sparse.csc_matrix((np.exp(-ND[:] * (1 / sig2)).flatten(), (XI[:].flatten(), NI[:].flatten())),
                               shape=(N, N)).todense()
    graphW += np.eye(N)

    return graphW, NNIndex


def gacCluster(distance_matrix, groupNumber, strDescr, K, a, z):

    print('--------------- Graph Structural Agglomerative Clustering ---------------------');

    # initialization

    print('---------- Building graph and forming initial clusters with l-links ---------');
    [graphW, NNIndex] = gacBuildDigraph(distance_matrix, K, a);
    # from adjacency matrix to probability transition matrix
    graphW = np.array((1. / np.sum(graphW, axis=1))) * np.array(graphW)  # row sum is 1
    initialClusters = gacNNMerge(distance_matrix, NNIndex)

    print('-------------------------- Zeta merging --------------------------');
    clusteredLabels = gacMerging(graphW, initialClusters, groupNumber, strDescr, z);

    return clusteredLabels


def predict_ac_mpi(feat, nClass, nSamples, nfeatures):
    K = 20
    a = 1
    z = 0.01

    distance_matrix = cdist(feat, feat) ** 2
    # path intergral
    label_pre = gacCluster(distance_matrix, nClass, 'path', K, a, z)

    return label_pre[:, 0]


def bestMap(L1, L2):
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)


def dataset_settings(dataset):
    if (dataset == 'MNIST-full') | (dataset == 'MNIST-test'):
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [0, 2]
        test_batch_size = 100
    elif dataset == 'FRGC':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [2, 2]
        test_batch_size = 1231
    elif dataset == 'COIL-20':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [2, 2]
        test_batch_size = 8
    elif dataset == 'YTF':
        kernel_sizes = [5, 4]
        strides = [2, 2]
        paddings = [2, 0]
        test_batch_size = 100

    return kernel_sizes, strides, paddings, test_batch_size


def create_result_dirs(output_path, file_name):
    if not os.path.exists(output_path):
        print('creating log folder')
        os.makedirs(output_path)
        try:
            os.makedirs(os.path.join(output_path, '../params'))
        except:
            pass
        func_file_name = os.path.basename(__file__)
        if func_file_name.split('.')[0] == 'pyc':
            func_file_name = func_file_name[:-1]
        functions_full_path = os.path.join(output_path, func_file_name)
        cmd = 'cp ' + func_file_name + ' "' + functions_full_path + '"'
        os.popen(cmd)
        run_file_full_path = os.path.join(output_path, file_name)
        cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)


class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(output_path + "log.txt", "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def kmeans(encoder_val_clean, y, nClusters, y_pred_prev=None, weight_initilization='k-means++', seed=42, n_init=40,
           max_iter=300):
    # weight_initilization = { 'kmeans-pca', 'kmean++', 'random', None }

    if weight_initilization == 'kmeans-pca':

        start_time = timeit.default_timer()
        pca = PCA(n_components=nClusters).fit(encoder_val_clean)
        kmeans_model = KMeans(init=pca.components_, n_clusters=nClusters, n_init=1, max_iter=300, random_state=seed)
        y_pred = kmeans_model.fit_predict(encoder_val_clean)

        centroids = kmeans_model.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))

        end_time = timeit.default_timer()

    elif weight_initilization == 'k-means++':

        start_time = timeit.default_timer()
        kmeans_model = KMeans(init='k-means++', n_clusters=nClusters, n_init=n_init, max_iter=max_iter, n_jobs=15,
                              random_state=seed)
        y_pred = kmeans_model.fit_predict(encoder_val_clean)

        D = 1.0 / euclidean_distances(encoder_val_clean, kmeans_model.cluster_centers_, squared=True)
        D **= 2.0 / (2 - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]

        centroids = kmeans_model.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))

        end_time = timeit.default_timer()

    print('k-means: \t nmi =', normalized_mutual_info_score(y, y_pred), '\t arc =', adjusted_rand_score(y, y_pred),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
          'K-means objective = {:.1f} '.format(kmeans_model.inertia_), '\t runtime =', end_time - start_time)

    if y_pred_prev is not None:
        print('Different Assignments: ', sum(y_pred == y_pred_prev), '\tbestMap: ', bestMap(y_pred, y_pred_prev),
              '\tdatapoints-bestMap*datapoints: ',
              encoder_val_clean.shape[0] - bestMap(y_pred, y_pred_prev) * encoder_val_clean.shape[0])

    return centroids, kmeans_model.inertia_, y_pred


def load_dataset(dataset_path):
    hf = h5py.File(dataset_path + '/data4torch.h5', 'r')
    X = np.asarray(hf.get('data'), dtype='float32')
    X_train = (X - np.float32(127.5)) / np.float32(127.5)
    y_train = np.asarray(hf.get('labels'), dtype='int32')
    return X_train, y_train


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], excerpt

# build_eml(encoder, n_out=num_clusters, W_initial=centroids)
def build_eml(input_var=None, n_out=None, W_initial=None):
    l_in = input_var

    if W_initial is None:
        l_out = lasagne.layers.DenseLayer(
            l_in, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Uniform(std=0.5, mean=0.5), b=lasagne.init.Constant(1))

    else:
        l_out = lasagne.layers.DenseLayer(
            l_in, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=W_initial, b=lasagne.init.Constant(0))

    return l_out


def build_depict(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                 dropouts=[0.1, 0.1, 0.1], kernel_sizes=[5, 5], strides=[2, 2],
                 paddings=[2, 2], hlayer_loss_param=0.1):
    # ENCODER
    l_e0 = lasagne.layers.DropoutLayer(
        lasagne.layers.InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = lasagne.layers.DropoutLayer(
        (lasagne.layers.Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                                    filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                                    nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01),
                                    W=lasagne.init.GlorotUniform())),
        p=dropouts[1])

    l_e2 = lasagne.layers.DropoutLayer(
        (lasagne.layers.Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                                    filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                                    nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01),
                                    W=lasagne.init.GlorotUniform())),
        p=dropouts[2])

    l_e2_flat = lasagne.layers.flatten(l_e2)

    l_e3 = lasagne.layers.DenseLayer(l_e2_flat, num_units=feature_map_sizes[2],
                                     nonlinearity=lasagne.nonlinearities.tanh)

    # DECODER
    l_d2_flat = lasagne.layers.DenseLayer(l_e3, num_units=l_e2_flat.output_shape[1],
                                          nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01))

    l_d2 = lasagne.layers.reshape(l_d2_flat,
                                  shape=[-1, l_e2.output_shape[1], l_e2.output_shape[2], l_e2.output_shape[3]])

    l_d1 = lasagne.layers.TransposedConv2DLayer(l_d2, num_filters=feature_map_sizes[0], stride=(strides[1], strides[1]),
                                        filter_size=(kernel_sizes[1], kernel_sizes[1]), crop=paddings[1],
                                        nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01))

    l_d0 = lasagne.layers.TransposedConv2DLayer(l_d1, num_filters=n_in[0], stride=(strides[0], strides[0]),
                                        filter_size=(kernel_sizes[0], kernel_sizes[0]), crop=paddings[0],
                                        nonlinearity=lasagne.nonlinearities.tanh)

    # Loss
    tar0 = input_var
    tar1 = lasagne.layers.get_output(l_e1, deterministic=True)
    tar2 = lasagne.layers.get_output(l_e2, deterministic=True)
    rec2 = lasagne.layers.get_output(l_d2)
    rec1 = lasagne.layers.get_output(l_d1)
    rec0 = lasagne.layers.get_output(l_d0)
    rec2_clean = lasagne.layers.get_output(l_d2, deterministic=True)
    rec1_clean = lasagne.layers.get_output(l_d1, deterministic=True)
    rec0_clean = lasagne.layers.get_output(l_d0, deterministic=True)

    loss0 = lasagne.objectives.squared_error(rec0, tar0)
    loss1 = lasagne.objectives.squared_error(rec1, tar1) * hlayer_loss_param
    loss2 = lasagne.objectives.squared_error(rec2, tar2) * hlayer_loss_param

    loss0_clean = lasagne.objectives.squared_error(rec0_clean, tar0)
    loss1_clean = lasagne.objectives.squared_error(rec1_clean, tar1) * hlayer_loss_param
    loss2_clean = lasagne.objectives.squared_error(rec2_clean, tar2) * hlayer_loss_param

    loss_recons = loss0.mean() + loss1.mean() + loss2.mean()
    loss_recons_clean = loss0_clean.mean() + loss1_clean.mean() + loss2_clean.mean()

    return l_e3, l_d0, loss_recons, loss_recons_clean


def train_depict_ae(dataset, X, y, input_var, decoder, encoder, loss_recons, loss_recons_clean, num_clusters, output_path,
                    batch_size=100, test_batch_size=100, num_epochs=1000, learning_rate=1e-4, verbose=1, seed=42,
                    continue_training=False):
    learning_rate_shared = theano.shared(lasagne.utils.floatX(learning_rate))
    params = lasagne.layers.get_all_params(decoder, trainable=True)
    # print("lasagne.layers.get_all_params(decoder, trainable=True), params.shape=",params.shape)
    updates = lasagne.updates.adam(loss_recons, params, learning_rate=learning_rate_shared)
    train_fn = theano.function([input_var], loss_recons, updates=updates)
    val_fn = theano.function([input_var], loss_recons_clean)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.10, random_state=42)
    best_val = np.inf
    last_update = 0

    # Load if pretrained weights are available.
    if os.path.isfile(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle')) & continue_training:
        with open(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle'),
                  "rb") as input_file:
            # print(input_file)
            # best_params = pickle.load(input_file, encoding='latin1')
            best_params = pickle.load(input_file)
            lasagne.layers.set_all_param_values(decoder, best_params)
    else:
        # TRAIN MODEL
        if verbose > 1:
            encoder_clean = lasagne.layers.get_output(encoder, deterministic=True)
            encoder_clean_function = theano.function([input_var], encoder_clean)

        for epoch in range(num_epochs + 1):
            train_err = 0
            num_batches = 0

            # Training
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, idx = batch
                train_err += train_fn(inputs)
                num_batches += 1

            validation_error = np.float32(val_fn(X_val))

            print("Epoch {} of {}".format(epoch + 1, num_epochs),
                  "\t  training loss:{:.6f}".format(train_err / num_batches),
                  "\t  validation loss:{:.6f}".format(validation_error))
            # if epoch % 10 == 0:
            last_update += 1
            if validation_error < best_val:
                last_update = 0
                print("new best error: ", validation_error)
                best_val = validation_error
                best_params_values = lasagne.layers.get_all_param_values(decoder)
                with open(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle'),
                          "wb") as output_file:
                    pickle.dump(best_params_values, output_file)
            if last_update > 100:
                break

            if (verbose > 1) & (epoch % 50 == 0):
                # Extract MdA features
                minibatch_flag = 1
                for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                    inputs, targets, idx = batch
                    minibatch_x = encoder_clean_function(inputs)
                    if minibatch_flag:
                        encoder_val_clean = minibatch_x
                        minibatch_flag = 0
                    else:
                        encoder_val_clean = np.concatenate((encoder_val_clean, minibatch_x), axis=0)

                kmeans(encoder_val_clean, y, num_clusters, seed=seed)

        last_params_values = lasagne.layers.get_all_param_values(decoder)
        with open(os.path.join(output_path, '../params/params_' + dataset + '_last.pickle'), "wb") as output_file:
            pickle.dump(params, output_file)
        with open(os.path.join(output_path, '../params/params_' + dataset + '_values_last.pickle'),
                  "wb") as output_file:
            pickle.dump(last_params_values, output_file)
        lasagne.layers.set_all_param_values(decoder, best_params_values)


def clustering(dataset, X, y, input_var, encoder, num_clusters, output_path, test_batch_size=100, seed=42,
               continue_training=False):
    encoder_clean = lasagne.layers.get_output(encoder, deterministic=True)
    encoder_clean_function = theano.function([input_var], encoder_clean)

    # Extract MdA features
    minibatch_flag = 1
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        inputs, targets, idx = batch
        minibatch_x = encoder_clean_function(inputs)
        if minibatch_flag:
            encoder_val_clean = minibatch_x
            minibatch_flag = 0
        else:
            encoder_val_clean = np.concatenate((encoder_val_clean, minibatch_x), axis=0)

    # Check kmeans results
    kmeans(encoder_val_clean, y, num_clusters, seed=seed)
    initial_time = timeit.default_timer()
    if (dataset == 'MNIST-full') | (dataset == 'FRGC') | (dataset == 'YTF') | (dataset == 'CMU-PIE'):
        # K-means on MdA Features
        centroids, inertia, y_pred = kmeans(encoder_val_clean, y, num_clusters, seed=seed)
        y_pred = (np.array(y_pred)).reshape(np.array(y_pred).shape[0], )
        y_pred = y_pred - 1
    else:
        # AC-PIC on MdA Features
        if os.path.isfile(os.path.join(output_path, '../params/pred' + dataset + '.pickle')) & continue_training:
            with open(os.path.join(output_path, '../params/pred' + dataset + '.pickle'), "rb") as input_file:
                # y_pred = pickle.load(input_file, encoding='latin1')
                y_pred = pickle.load(input_file)
        else:
            try:
                import matlab.engine
                eng = matlab.engine.start_matlab()
                eng.addpath(eng.genpath('matlab'))
                targets_init = eng.predict_ac_mpi(
                    matlab.double(
                        encoder_val_clean.reshape(encoder_val_clean.shape[0] * encoder_val_clean.shape[1]).tolist()),
                    num_clusters, encoder_val_clean.shape[0], encoder_val_clean.shape[1])
                y_pred = (np.array(targets_init)).reshape(np.array(targets_init).shape[0], )
                eng.quit()
                y_pred = y_pred - 1
            except:
                y_pred = predict_ac_mpi(encoder_val_clean, num_clusters, encoder_val_clean.shape[0],
                                        encoder_val_clean.shape[1])
            with open(os.path.join(output_path, '../params/pred' + dataset + '.pickle'), "wb") as output_file:
                pickle.dump(y_pred, output_file)

        final_time = timeit.default_timer()
        print('AC-PIC: \t nmi =  ', normalized_mutual_info_score(y, y_pred),
              '\t arc = ', adjusted_rand_score(y, y_pred),
              '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
              '\t time taken = {:.4f}'.format(final_time - initial_time))
        centroids_acpic = np.zeros(shape=(num_clusters, encoder_val_clean.shape[1]))
        for i in range(num_clusters):
            centroids_acpic[i] = encoder_val_clean[y_pred == i].mean(axis=0)

        centroids = centroids_acpic.T
        centroids = centroids_acpic / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))

    return np.int32(y_pred), np.float32(centroids)


def train_depict(dataset, X, y, input_var, decoder, encoder, loss_recons, num_clusters, y_pred, output_path,
                 batch_size=100, test_batch_size=100, num_epochs=1000, learning_rate=1e-4, prediction_status='soft',
                 rec_mult=1, clus_mult=1, verify_mult=1, centroids=None, init_flag=0, continue_training=False):
    ######################
    #   ADD RLC TO MdA   #
    ######################

    initial_time = timeit.default_timer()
    rec_lambda = theano.shared(lasagne.utils.floatX(rec_mult))
    clus_lambda = theano.shared(lasagne.utils.floatX(clus_mult))
    pred_normalizition_flag = 1
    num_batches = X.shape[0] // batch_size

    if prediction_status == 'soft':
        target_var = T.matrix('minibatch_out')
        verify_pos_coef = T.matrix('verify_pos_coef')
        verify_neg_coef = T.matrix('verify_neg_coef')
        verify_coef = T.matrix('verify_coef')
        target_init = T.ivector('kmeans_out')
        # verify_coef_init = T.matrix('verify_coef_init')
    elif prediction_status == 'hard':
        target_var = T.ivector('minibatch_out')
        target_val = T.vector()

    network2 = build_eml(encoder, n_out=num_clusters, W_initial=centroids)
    network_prediction_noisy = lasagne.layers.get_output(network2, input_var, deterministic=False)
    network_prediction_clean = lasagne.layers.get_output(network2, input_var, deterministic=True)
    encoder_clean = lasagne.layers.get_output(encoder, input_var, deterministic=True)
    encoder_noise = lasagne.layers.get_output(encoder, input_var, deterministic=False)

    loss_clus_init = lasagne.objectives.categorical_crossentropy(network_prediction_noisy, target_init).mean()
    p1 = T.sum(encoder_noise**2, axis=1, dtype='float32', keepdims=True)
    p12 = T.dot(encoder_noise,encoder_noise.T)
    loss_verifi = T.add(T.add(p1,(-2.)*p12),p1.T)
    # loss_verifi, updates = theano.scan(lambda x, y:np.abs(x-y).sum(axis = 1),
    #     sequences = [encoder_noise],
    #     non_sequences = encoder_noise)

    verify_neg_input =  lasagne.layers.InputLayer((batch_size,batch_size))
    verify_neg_net = lasagne.layers.ScaleLayer(verify_neg_input, scales = lasagne.init.Constant(50),shared_axes=(0,1))
    # print(verify_neg_net.b.get_value().shape)
    verify_m = lasagne.layers.get_output(verify_neg_net, verify_neg_coef, deterministic=True)
    # verify_m = theano.shared(lasagne.utils.floatX(0.1))
    # verify_neg = loss_verifi - verify_m


    params_init = lasagne.layers.get_all_params([decoder, network2], trainable=True)

    if prediction_status == 'soft':
        loss_clus = lasagne.objectives.categorical_crossentropy(network_prediction_noisy,
                                                                target_var)
    elif prediction_status == 'hard':
        loss_clus = target_val * lasagne.objectives.categorical_crossentropy(network_prediction_noisy, target_var)

    loss_clus = clus_lambda * loss_clus.mean()
    loss_recons = rec_lambda * loss_recons
    loss_verify = verify_mult * T.mean(T.maximum((-verify_neg_coef)*loss_verifi+verify_m, 0) * verify_coef)
    loss = loss_recons + loss_clus + loss_verify
    # params2 = lasagne.layers.get_all_params([decoder, network2, verify_neg_net], trainable=True)
    params2 = lasagne.layers.get_all_params([decoder, network2], trainable=True)
    updates = lasagne.updates.adam(
        loss, params2, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var,verify_pos_coef,verify_neg_coef,verify_coef],
                               [loss, loss_recons, loss_clus, loss_verify], updates=updates,on_unused_input='ignore')

    loss_clus_init = clus_lambda * loss_clus_init
    loss_init = loss_clus_init + loss_recons#+ loss_verify_init
    updates_init = lasagne.updates.adam(
        loss_init, params_init, learning_rate=learning_rate)
    train_fn_init = theano.function([input_var, target_init],
                                    [loss_init, loss_recons, loss_clus_init], updates=updates_init)

    test_fn = theano.function([input_var], network_prediction_clean)
    final_time = timeit.default_timer()

    print("\n...Start DEPICT initialization")
    if init_flag:
        if os.path.isfile(os.path.join(output_path, '../params/weights' + dataset + '.pickle')) & continue_training:
            with open(os.path.join(output_path, '../params/weights' + dataset + '.pickle'),
                      "rb") as input_file:
                # weights = pickle.load(input_file, encoding='latin1')
                weights = pickle.load(input_file)
                lasagne.layers.set_all_param_values([decoder, network2], weights)
        else:
            X_train, X_val, y_train, y_val, y_pred_train, y_pred_val = train_test_split(
                X, y, y_pred, stratify=y, test_size=0.10, random_state=42)
            last_update = 0
            # Initilization
            y_targ_train = np.copy(y_pred_train)
            y_targ_val = np.copy(y_pred_val)
            y_val_prob = test_fn(X_val)
            y_val_pred = np.argmax(y_val_prob, axis=1)
            val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)
            best_val = 0.0
            print('initial val nmi: ', val_nmi)
            print('initial val acc: ', bestMap(y_targ_val, y_val_pred))
            best_params_values = lasagne.layers.get_all_param_values([decoder, network2])
            for epoch in range(1000):
                train_err, val_err = 0, 0
                lossre_train, lossre_val = 0, 0
                losspre_train, losspre_val = 0, 0
                lossverify_train = 0
                num_batches_train = 0
                for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                    minibatch_inputs, targets, idx = batch
                    target_matrix_init = np.float32(np.eye(num_clusters)[y_targ_train[idx]])
                    verify_coef_init = (np.dot(target_matrix_init,target_matrix_init.T)-0.5)*2-np.eye(target_matrix_init.shape[0])
                    verify_m_coef = 1.- np.dot(target_matrix_init,target_matrix_init.T)
                    minibatch_error, lossrec, losspred= train_fn_init(minibatch_inputs, np.int32(y_targ_train[idx]))
                    train_err += minibatch_error
                    lossre_train += lossrec
                    losspre_train += losspred
                    # lossverify_train += lossverify
                    num_batches_train += 1

                y_val_prob = test_fn(X_val)
                y_val_pred = np.argmax(y_val_prob, axis=1)

                y_pred = np.zeros(X.shape[0])
                for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                    minibatch_inputs, targets, idx = batch
                    minibatch_prob = test_fn(minibatch_inputs)
                    minibatch_pred = np.argmax(minibatch_prob, axis=1)
                    y_pred[idx] = minibatch_pred

                val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)

                print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                      '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                      '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
                      '\t loss= {:.10f}'.format(train_err / num_batches_train),
                      '\t loss_reconstruction= {:.10f}'.format(lossre_train / num_batches_train),
                      '\t loss_prediction= {:.10f}'.format(losspre_train / num_batches_train),
                      # '\t loss_verification= {:.10f}'.format(lossverify_train / num_batches_train),
                      '\t val nmi = {:.4f}  '.format(val_nmi))
                last_update += 1
                if val_nmi > best_val:
                    last_update = 0
                    print("new best val nmi: ", val_nmi)
                    best_val = val_nmi
                    best_params_values = lasagne.layers.get_all_param_values([decoder, network2])
                    # if (losspre_val / num_batches_val) < 0.2:
                    #     break

                if last_update > 5:
                    break

            lasagne.layers.set_all_param_values([decoder, network2], best_params_values)
            with open(os.path.join(output_path, '../params/weights' + dataset + '.pickle'), "wb") as output_file:
                pickle.dump(lasagne.layers.get_all_param_values([decoder, network2]), output_file)

    # Epoch 0
    print("\n...Start DEPICT training")
    y_prob = np.zeros((X.shape[0], num_clusters))
    y_prob_prev = np.zeros((X.shape[0], num_clusters))
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        minibatch_inputs, targets, idx = batch
        minibatch_prob = test_fn(minibatch_inputs)
        y_prob[idx] = minibatch_prob

    y_prob_max = np.max(y_prob, axis=1)
    if pred_normalizition_flag:
        cluster_frequency = np.sum(y_prob, axis=0)
        y_prob = y_prob ** 2 / cluster_frequency
        # y_prob = y_prob / np.sqrt(cluster_frequency)
        y_prob = np.transpose(y_prob.T / np.sum(y_prob, axis=1))
    y_pred = np.argmax(y_prob, axis=1)

    print('epoch: 0', '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)))
    if os.path.isfile(os.path.join(output_path, '../params/rlc' + dataset + '.pickle')) & continue_training:
        with open(os.path.join(output_path, '../params/rlc' + dataset + '.pickle'),
                  "rb") as input_file:
            weights = pickle.load(input_file, encoding='latin1')
            lasagne.layers.set_all_param_values([decoder, network2], weights)
    else:
        for epoch in range(num_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            lossre = 0
            losspre = 0
            lossver = 0

            for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
                minibatch_inputs, targets, idx = batch

                # M_step
                if prediction_status == 'hard':
                    minibatch_err, lossrec, losspred, lossverify= train_fn(minibatch_inputs,
                                                                np.ndarray.astype(y_pred[idx], 'int32'),
                                                                np.ndarray.astype(y_prob_max[idx],
                                                                                  'float32'))
                elif prediction_status == 'soft':
                    temp = np.eye(num_clusters)[y_pred[idx]]
                    # print(y_prob_max[idx].shape)
                    y_prob_max_batch = np.expand_dims(y_prob_max[idx], axis=1)
                    coef = np.dot(y_prob_max_batch,y_prob_max_batch.T)
                    # print(coef)
                    # verify_coef = (np.dot(temp,temp.T)-0.5)*2-np.eye(temp.shape[0])
                    verify_pos_coef = np.dot(temp,temp.T)-np.eye(temp.shape[0])
                    verify_neg_coef = 1.0 - np.dot(temp,temp.T)
                    # print(verify_coef)
                    # verify_m_coef = 1.0 - np.dot(temp,temp.T)
                    # print(verify_m_coef)
                    # print(verify_coef)
                    # minibatch_err, lossrec, losspred ,lossverify= train_fn(minibatch_inputs,
                    #                                             np.ndarray.astype(y_prob[idx], 'float32'),
                    #                                             np.ndarray.astype(verify_coef*coef, 'float32'),
                    #                                             np.ndarray.astype(verify_m_coef*coef, 'float32'))
                    minibatch_err, lossrec, losspred ,lossverify= train_fn(minibatch_inputs,
                                                                np.ndarray.astype(y_prob[idx], 'float32'),
                                                                np.ndarray.astype(verify_pos_coef, 'float32'),
                                                                np.ndarray.astype(verify_neg_coef, 'float32'),
                                                                np.ndarray.astype(coef, 'float32'))

                minibatch_prob = test_fn(minibatch_inputs)
                y_prob[idx] = minibatch_prob
                train_err += minibatch_err
                lossre += lossrec
                losspre += losspred
                lossver += lossverify

            y_prob_max = np.max(y_prob, axis=1)
            # print(y_prob_max)
            if pred_normalizition_flag:
                cluster_frequency = np.sum(y_prob, axis=0)  # avoid unbalanced assignment
                y_prob = y_prob ** 2 / cluster_frequency
                # y_prob = y_prob / np.sqrt(cluster_frequency)
                y_prob = np.transpose(y_prob.T / np.sum(y_prob, axis=1))
            y_pred = np.argmax(y_prob, axis=1)

            # print('mse: ', mean_squared_error(y_prob, y_prob_prev))

            if mean_squared_error(y_prob, y_prob_prev) < 1e-7:
                with open(os.path.join(output_path, '../params/rlc' + dataset + '.pickle'), "wb") as output_file:
                    pickle.dump(lasagne.layers.get_all_param_values([decoder, network2]), output_file)
                break
            y_prob_prev = np.copy(y_prob)

            print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                  '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                  '\t acc = {:.4f} '.format(bestMap(y, y_pred)), '\t loss= {:.10f}'.format(train_err / num_batches),
                  '\t loss_recons= {:.10f}'.format(lossre / num_batches),
                  '\t loss_pred= {:.10f}'.format(losspre / num_batches),
                  '\t loss_verify= {:.10f}'.format(lossver / num_batches))

    # test
    y_pred = np.zeros(X.shape[0])
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        minibatch_inputs, targets, idx = batch
        minibatch_prob = test_fn(minibatch_inputs)
        minibatch_pred = np.argmax(minibatch_prob, axis=1)
        y_pred[idx] = minibatch_pred

    print('final: ', '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)))


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated

_Datasets = collections.namedtuple('_Datasets', ['train', 'validation', 'test'])

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


@deprecated(None, 'Please use tf.data to implement this functionality.')
def _extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


@deprecated(None, 'Please use tf.one_hot on tensors.')
def _dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


@deprecated(None, 'Please use tf.data to implement this functionality.')
def _extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return _dense_to_one_hot(labels, num_classes)
    return labels


class _DataSet(object):
  """Container class for a _DataSet (deprecated).
  THIS CLASS IS DEPRECATED.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/_DataSet.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a _DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    Args:
      images: The images
      labels: The labels
      fake_data: Ignore inages and labels, use fake data.
      one_hot: Bool, return the labels as one hot vectors (if True) or ints (if
        False).
      dtype: Output image dtype. One of [uint8, float32]. `uint8` output has
        range [0,255]. float32 output has range [0,1].
      reshape: Bool. If True returned images are returned flattened to vectors.
      seed: The random seed to use.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)
             ], [fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part),
                               axis=0), numpy.concatenate(
                                   (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


@deprecated(None, 'Please write your own downloading logic.')
def _maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.
  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.
  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not gfile.Exists(filepath):
    urllib.request.urlretrieve(source_url, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


@deprecated(None, 'Please use alternatives such as:'
            ' tensorflow_datasets.load(\'mnist\')')
def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  if fake_data:

    def fake():
      return _DataSet([], [],
                      fake_data=True,
                      one_hot=one_hot,
                      dtype=dtype,
                      seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return _Datasets(train=train, validation=validation, test=test)

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  train_images_file = 'train-images-idx3-ubyte.gz'
  train_labels_file = 'train-labels-idx1-ubyte.gz'
  test_images_file = 't10k-images-idx3-ubyte.gz'
  test_labels_file = 't10k-labels-idx1-ubyte.gz'

  local_file = _maybe_download(train_images_file, train_dir,
                               source_url + train_images_file)
  with gfile.Open(local_file, 'rb') as f:
    train_images = _extract_images(f)

  local_file = _maybe_download(train_labels_file, train_dir,
                               source_url + train_labels_file)
  with gfile.Open(local_file, 'rb') as f:
    train_labels = _extract_labels(f, one_hot=one_hot)

  local_file = _maybe_download(test_images_file, train_dir,
                               source_url + test_images_file)
  with gfile.Open(local_file, 'rb') as f:
    test_images = _extract_images(f)

  local_file = _maybe_download(test_labels_file, train_dir,
                               source_url + test_labels_file)
  with gfile.Open(local_file, 'rb') as f:
    test_labels = _extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'.format(
            len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = _DataSet(train_images, train_labels, **options)
  validation = _DataSet(validation_images, validation_labels, **options)
  test = _DataSet(test_images, test_labels, **options)

  return _Datasets(train=train, validation=validation, test=test)

import argparse
import socket
__file__='result1'

############################## settings ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42)
parser.add_argument('--dataset', default='USPS')
parser.add_argument('--continue_training', action='store_true', default=False)
parser.add_argument('--datasets_path', default='/datasets/')
parser.add_argument('--feature_map_sizes', default=[50, 50, 10])
parser.add_argument('--dropouts', default=[0.1, 0.1, 0.0])
parser.add_argument('--batch_size', default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', default=4000)
parser.add_argument('--reconstruct_hyperparam', default=1.)
parser.add_argument('--cluster_hyperparam', default=1.)

parser.add_argument('--architecture_visualization_flag', default=1)
parser.add_argument('--loss_acc_plt_flag', default=1)
parser.add_argument('--verbose', default=2)
args = parser.parse_args(args=[])

############################## Logging ##############################
output_path = '/results/'+os.path.basename(__file__).split('.')[0] + '/' + time.strftime("%d-%m-%Y_") + \
              time.strftime("%H:%M:%S") + '_' + args.dataset + '_' + socket.gethostname()
pyscript_name = os.path.basename(__file__)
create_result_dirs(output_path, pyscript_name)
sys.stdout = Logger(output_path)
print(args)
print('----------')
#print(sys.argv)

# fixed random seeds
seed = args.seed
np.random.seed(args.seed)
rng = np.random.RandomState(seed)
theano_rng = MRG_RandomStreams(seed)
lasagne.random.set_rng(np.random.RandomState(seed))
learning_rate = args.learning_rate
dataset = args.dataset
datasets_path = args.datasets_path
dropouts = args.dropouts
feature_map_sizes = args.feature_map_sizes
num_epochs = args.num_epochs
batch_size = args.batch_size
cluster_hyperparam = args.cluster_hyperparam
reconstruct_hyperparam = args.reconstruct_hyperparam
verbose = args.verbose

############################## Load Data  ##############################
#X, y = load_dataset(datasets_path + dataset)
mnist = read_data_sets(os.path.join(datasets_path,"mnist"), one_hot=False)

mnist_images_train = mnist.train.images
mnist_images_val = mnist.validation.images
mnist_images_test = mnist.test.images
mnist_images_full = np.vstack((mnist_images_train,mnist_images_val,mnist_images_test))


mnist_labels_train = mnist.train.labels
mnist_labels_val = mnist.validation.labels
mnist_labels_test = mnist.test.labels
mnist_labels_full =np.concatenate([mnist_labels_train, mnist_labels_val, mnist_labels_test])

X = (mnist_images_test-0.5)*2.
# print np.amin(X)
# print np.amax(X)
y = mnist_labels_test
shape = X.shape
# print shape
X = X.reshape(shape[0],1,28,28)
X = X[0:10300]
y = y[0:10300]
#print X.shape
# print X[0]

#print np.amin(X)
#print np.amax(X)
# shape = X.shape
#X = X.reshape(shape[0],3,55,55)

num_clusters = len(np.unique(y))
num_samples = len(y)
dimensions = [X.shape[1], X.shape[2], X.shape[3]]
print('dataset: %s \tnum_samples: %d \tnum_clusters: %d \tdimensions: %s'
      % (dataset, num_samples, num_clusters, str(dimensions)))

feature_map_sizes[-1] = num_clusters
input_var = T.tensor4('inputs')
kernel_sizes, strides, paddings, test_batch_size = dataset_settings(dataset)
print(
    '\n... build DEPICT model...\nfeature_map_sizes: %s \tdropouts: %s \tkernel_sizes: %s \tstrides: %s \tpaddings: %s'
    % (str(feature_map_sizes), str(dropouts), str(kernel_sizes), str(strides), str(paddings)))

##############################  Build DEPICT Model  ##############################
encoder, decoder, loss_recons, loss_recons_clean = build_depict(input_var, n_in=dimensions,
                                                                feature_map_sizes=feature_map_sizes,
                                                                dropouts=dropouts, kernel_sizes=kernel_sizes,
                                                                strides=strides,
                                                                paddings=paddings)

############################## Pre-train DEPICT Model   ##############################
print("\n...Start AutoEncoder training...")
initial_time = timeit.default_timer()
train_depict_ae(dataset, X, y, input_var, decoder, encoder, loss_recons, loss_recons_clean, num_clusters, output_path,
                batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                verbose=verbose, seed=seed, continue_training=args.continue_training)

############################## Clustering Pre-trained DEPICT Features  ##############################
y_pred, centroids = clustering(dataset, X, y, input_var, encoder, num_clusters, output_path,
                               test_batch_size=test_batch_size, seed=seed, continue_training=args.continue_training)

############################## Train DEPICT Model  ##############################
train_depict(dataset, X, y, input_var, decoder, encoder, loss_recons, num_clusters, y_pred, output_path,
             batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs,
             learning_rate=learning_rate, rec_mult=reconstruct_hyperparam, clus_mult=cluster_hyperparam,
             centroids=centroids, continue_training=args.continue_training)

final_time = timeit.default_timer()

print('Total time for ' + dataset + ' was: ' + str((final_time - initial_time)))
