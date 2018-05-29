#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import csv

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn.decomposition import FactorAnalysis, PCA, TruncatedSVD
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def linearize_unit(units, keys):
    return list(map(lambda prop: float(prop[1]), filter(lambda prop: prop[0] in keys, units.items())))


def cluster_info(cluster, linearize_units, units):
    predictions = cluster.fit_predict(linearize_units)
    predictCluster = {}
    predictIdToType = {}
    for idx, unit in enumerate(units):
        group = predictions[idx]
        if predictCluster.get(group) is None:
            predictCluster[group] = []
        predictCluster[group].append(unit.get('id'))
        predictIdToType[unit.get('id')] = group
    return {'predictCluster': predictCluster, 'predictIdToType': predictIdToType}


def clusterize(filename: str, keys: list):
    """
    :param keys:
    :param filename:
    :type filename: str
    :return:
    """
    # Список с загруженными значениями
    mcu = []
    # Список параметров
    mcu_params = []
    # Список значений, которые принимает параметр
    mcu_params_vars = {}
    with open(os.path.join('data', filename), 'r') as csv_file:
        reader = csv.reader(csv_file)
        for (num, line) in enumerate(reader):
            if num == 0:
                mcu_params = line
                for value in mcu_params:
                    mcu_params_vars[value] = set()
            else:
                unit = {'id': num}
                for idx, value in enumerate(line):
                    # Замена пустых строк и несуществующих значений на нули
                    if mcu_params[idx] in keys and value is '':
                        value = 0
                    unit[mcu_params[idx]] = value
                    mcu_params_vars[mcu_params[idx]].add(value)
                mcu.append(unit)

    new_mcu = []
    for unit in mcu:
        new_unit = {}
        for param, value in unit.items():
            if param in keys or param is 'id':
                new_unit[param] = value
        new_mcu.append(new_unit)
    mcu = new_mcu

    X = np.array(list(map(lambda unit: linearize_unit(unit, keys), mcu)))

    # Кластеризация (K-средних)
    kmeans = KMeans(n_clusters=7).fit(X)
    kmeansCluster = cluster_info(kmeans, X, mcu)

    # Кластеризация (AffinityPropagation)
    affinity = AffinityPropagation().fit(X)
    affinityCluster = cluster_info(affinity, X, mcu)

    # Кластеризация (DBSCAN)
    dbscan = DBSCAN().fit(X)
    dbscanCluster = cluster_info(dbscan, X, mcu)

    # Сравнение количества элементов по кластерам
    print('KMeans:')
    print(list(map(lambda cluster: (cluster[0], len(cluster[1])), kmeansCluster.get('predictCluster').items())))
    print('AffinityCluster:')
    print(list(map(lambda cluster: (cluster[0], len(cluster[1])), affinityCluster.get('predictCluster').items())))
    print('DBSCAN Cluster:')
    print(list(map(lambda cluster: (cluster[0], len(cluster[1])), dbscanCluster.get('predictCluster').items())))

    # Снижение размерности пространства:
    kmeansClassification = list(map(lambda idToType: idToType[1], kmeansCluster.get('predictIdToType').items()))
    affinityClassification = list(map(lambda idToType: idToType[1], affinityCluster.get('predictIdToType').items()))
    dbscanClassification = list(map(lambda idToType: idToType[1], dbscanCluster.get('predictIdToType').items()))

    # Снижение размерности пространства: TruncatedSVD
    X_truncated = TruncatedSVD(n_components=3).fit_transform(X)

    trunk_fig_kmean = plt.figure()
    ax = trunk_fig_kmean.add_subplot(111, projection='3d')
    ax.scatter(X_truncated[:, 0], X_truncated[:, 1], X_truncated[:, 2], c=kmeansClassification)
    ax.set_title("Кластеризация k-средних: TruncatedSVD  (3d) of ({}d)".format(len(keys)))

    trunk_fig_affinity = plt.figure()
    ax = trunk_fig_affinity.add_subplot(111, projection='3d')
    ax.scatter(X_truncated[:, 0], X_truncated[:, 1], X_truncated[:, 2], c=affinityClassification)
    ax.set_title("Кластеризация Affinity Propagation: TruncatedSVD (3d) of ({}d)".format(len(keys)))

    trunk_fig_dbscan = plt.figure()
    ax = trunk_fig_dbscan.add_subplot(111, projection='3d')
    ax.scatter(X_truncated[:, 0], X_truncated[:, 1], X_truncated[:, 2], c=dbscanClassification)
    ax.set_title("Кластеризация DBSCAN: TruncatedSVD (3d) of ({}d)".format(len(keys)))

    plt.show()

    # Снижение размерности пространства: Pricipal Component Analysis
    X_factor = PCA(n_components=3).fit_transform(X)

    pca_fig_kmean = plt.figure()
    ax = pca_fig_kmean.add_subplot(111, projection='3d')
    ax.scatter(X_factor[:, 0], X_factor[:, 1], X_factor[:, 2], c=kmeansClassification)
    ax.set_title("Кластеризация k-средних: PCA reduction (3d) of ({}d)".format(len(keys)))

    pca_fig_affinity = plt.figure()
    ax = pca_fig_affinity.add_subplot(111, projection='3d')
    ax.scatter(X_factor[:, 0], X_factor[:, 1], X_factor[:, 2], c=affinityClassification)
    ax.set_title("Кластеризация Affinity Clustering: PCA reduction (3d) of ({}d)".format(len(keys)))

    pca_fig_dbscan = plt.figure()
    ax = pca_fig_dbscan.add_subplot(111, projection='3d')
    ax.scatter(X_factor[:, 0], X_factor[:, 1], X_factor[:, 2], c=dbscanClassification)
    ax.set_title("Кластеризация DBSCAN: PCA reduction (3d) of ({}d)".format(len(keys)))

    plt.show()

    # Снижение размерности пространства: Factor Analysis
    X_factor = FactorAnalysis(n_components=3).fit_transform(X)

    factor_fig_kmean = plt.figure()
    ax = factor_fig_kmean.add_subplot(111, projection='3d')
    ax.scatter(X_factor[:, 0], X_factor[:, 1], X_factor[:, 2], c=kmeansClassification)
    ax.set_title("Кластеризация k-средних: Factor Analysis (3d) of ({}d)".format(len(keys)))

    factor_fig_affinity = plt.figure()
    ax = factor_fig_affinity.add_subplot(111, projection='3d')
    ax.scatter(X_factor[:, 0], X_factor[:, 1], X_factor[:, 2], c=affinityClassification)
    ax.set_title("Кластеризация Affinity Clustering: Factor  reduction (3d) of ({}d)".format(len(keys)))

    factor_fig_dbscan = plt.figure()
    ax = factor_fig_dbscan.add_subplot(111, projection='3d')
    ax.scatter(X_factor[:, 0], X_factor[:, 1], X_factor[:, 2], c=dbscanClassification)
    ax.set_title("Кластеризация DBSCAN: Factor Analysis (3d) of ({}d)".format(len(keys)))

    plt.show()
