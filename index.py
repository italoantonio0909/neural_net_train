from sklearn.datasets import make_circles
import argparse
import numpy
from matplotlib import pyplot as plt
import pandas
import os


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


def train(nn, lr=0.5, train: bool = True):
    # Ir hacia adelante
    # Realizar sumas ponderadas y funciones de activación
    out = [(None, X)]

    data = []

    for l, layer in enumerate(nn):
        z = out[-1][1] @ nn[l].w + nn[l].b
        a = nn[l].act_f(z)
        out.append((z, a))

        data.append((z, a))

    columns = ['Sum', 'Activation']
    save_csv(filename='train_fordwardpass.csv', columns=columns, data=data)

    if train:
        # Ir hacia atras y calcular deltas
        # Backpropagation

        deltas = []

        for l in reversed(range(0, len(nn))):
            z = out[l+1][0]
            a = out[l+1][1]

            # Cálculo de delta en función a la última capa
            if l == len(nn)-1:
                deltas.insert(0, error_cost_derivate(
                    a, Y) * nn[l].act_f_derivate(a))

            else:
                # Cálculo de delta en función de capas previas
                deltas.insert(0, deltas[0] @ nn[l].w.T *
                              nn[l].act_f_derivate(a))


if __name__ == '__main__':
    args = cli()
    samples = int(args.samples)

    X, Y = make_circles(n_samples=samples, factor=0.5, noise=0.05)

    topology = [2, 4, 8, 1]
    nn = create_neural_net(topology, sigmoide, sigmoide_derivate)

    train(nn=nn)
