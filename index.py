import time

import numpy
import scipy
from IPython.display import clear_output
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
import argparse
import os
import pandas


def cli():
    args = argparse.ArgumentParser(
        prog='Neural Net Train console',
    )
    args.add_argument('samples', help='Number samples')
    return args.parse_args()


class NeuralLayer():
    def __init__(self, n_conn, n_neural, act_f, act_f_derivate):
        self.act_f = act_f
        self.act_f_derivate = act_f_derivate

        # Parámetro de BAIAS
        self.b = numpy.random.rand(1, n_neural) * 2-1

        # Parámetro de Peso
        self.w = numpy.random.rand(n_conn, n_neural) * 2-1


def get_directory(*, filename: str):
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return os.path.join(data_dir, filename)


def save_csv(*, filename, columns, data):
    filename = get_directory(filename=filename)
    df = pandas.DataFrame(data, columns=columns)
    df.to_csv(filename)


def create_neural_net(topology, act_f, act_f_derivate):

    nn = []
    data = []

    # Topology -> [2, 4, 8, 1]
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(
            topology[l], topology[l+1], act_f, act_f_derivate))

        data.append((topology[l], topology[l+1]))

    # save data
    columns = ['Number connections', 'Number Neural']
    save_csv(filename='neural_net.csv', columns=columns, data=data)

    return nn


def sigmoide(x): return 1 / (1+numpy.e ** (-x))
def sigmoide_derivate(x): return x * (1-x)


def error_cost(Yp, Yr): return numpy.mean((Yp-Yr)**2)
def error_cost_derivate(Yp, Yr): return Yp-Yr


def train(*, nn, lr=0.05, X, Y, train: bool = True, l2_cost, l2_cost_derivate):
    # Ir hacia adelante
    # Realizar sumas ponderadas y funciones de activación
    out = [(None, X)]

    data = []

    for l, layer in enumerate(nn):
        z = out[-1][1] @ nn[l].w + nn[l].b
        a = nn[l].act_f(z)
        out.append((z, a))

        data.append((z, a))

    # columns = ['Sum', 'Activation']
    # save_csv(filename='train_fordwardpass.csv', columns=columns, data=data)

    if train:
        # Ir hacia atras y calcular deltas
        # Backpropagation

        deltas = []

        for l in reversed(range(0, len(nn))):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(nn)-1:
                # Cálculo de delta en función a la última capa
                deltas.insert(0, l2_cost_derivate(
                    a, Y) * nn[l].act_f_derivate(a))

            else:
                # Cálculo de delta en función de capas previas
                deltas.insert(0, deltas[0] @ _w.T *
                              nn[l].act_f_derivate(a))

            # Primer iteración
            # Iteración number 2 _w vale
            # [[-0.88814489]
            #  [ 0.28757434]
            #  [ 0.19901192]
            #  [-0.61129388]
            #  [ 0.1513037 ]
            #  [-0.74521616]
            #  [-0.31054925]
            #  [ 0.0257846 ]]

            _w = nn[l].w

            # Descenso del gradiente
            # Mejora del parámetro de BAIAS
            nn[l].b = nn[l].b - \
                numpy.mean(deltas[0], axis=0, keepdims=True) * lr
            # Mejora del parámetro de Pesos
            nn[l].w = nn[l].w - out[l][1].T @ deltas[0] * lr

    return out[-1][1]


def deep_train(*, nn, l2_cost, l2_cost_derivate, X, Y):
    # Entrenamiento profundo de la red neuronal
    loss = []
    for i in range(1000):
        pY = train(nn=nn, X=X, Y=Y, l2_cost=error_cost,
                   l2_cost_derivate=error_cost_derivate)

        if i % 25 == 0:
            loss.append(l2_cost(pY, Y))

            res = 50

            _x0 = numpy.linspace(-1.5, 1.5, res)
            _x1 = numpy.linspace(-1.5, 1.5, res)

            _y = numpy.zeros((res, res))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _y[i0, i1] = train(nn=nn, X=numpy.array(
                        [[x0, x1]]), Y=Y, l2_cost=l2_cost, l2_cost_derivate=l2_cost_derivate, train=False)[0][0]

            plt.pcolormesh(_x0, _x1, _y, cmap='coolwarm')
            plt.axis('equal')

            plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c='skyblue')
            plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c='salmon')

            clear_output(wait=True)
            plt.show()
            plt.plot(range(len(loss)), loss)
            plt.show()
            time.sleep(0.5)


if __name__ == '__main__':
    args = cli()
    samples = int(args.samples)

    X, Y = make_circles(n_samples=samples, factor=0.5, noise=0.05)
    Y = Y[:, numpy.newaxis]

    topology = [2, 4, 8, 1]
    nn = create_neural_net(topology, sigmoide, sigmoide_derivate)

    deep_train(nn=nn, l2_cost=error_cost,
               l2_cost_derivate=error_cost_derivate, X=X, Y=Y)
