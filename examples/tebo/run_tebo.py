import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
import json
import argparse
import numpy as np
import ConfigSpace

from functools import partial

from tebo.tebo_lvgp import initialize_model, make_new_suggestion, evaluate_augment_dataset, \
    initialize_model_from_existing, save_plot_target_task, save_model, normalize_Y, make_initial_suggestion
from hpolib.benchmarks.surrogates.paramnet import SurrogateReducedParamNetTime

import mxnet as mx

ctx = mx.gpu()
mx.context.Context.default_ctx = ctx

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?')
parser.add_argument('--main_task', default="mnist", type=str, nargs='?')
parser.add_argument('--seed', default=0, type=int, nargs='?')
parser.add_argument('--output_path', default="./", type=str, nargs='?')
parser.add_argument('--n_init', default=50, type=int, nargs='?')
parser.add_argument('--n_iters', default=3, type=int, nargs='?')
parser.add_argument('--start_from_bo', action='store_true', default=False)
parser.add_argument('--bo_data', default="./", type=str, nargs='?')

args = parser.parse_args()


tasks = ["adult", "higgs", "letter", "optdigits", "poker", "vehicle", "mnist"]

# target task needs to always the last index
tasks.remove(args.main_task)
tasks.append(args.main_task)

funcs = []
for t in tasks:
    f = SurrogateReducedParamNetTime(t)
    funcs.append(f)


cs = funcs[0].get_configuration_space()  # Configspace is always the same

Q = len(cs.get_hyperparameters())  # the input dimensionality
Q_h = 2  # the dimensionality of latent space
M_init = (50, 100)  # the number of inducing points used for initialization
nSamples = 10  # the number of samples for variational inference

target_task = len(tasks) - 1


def objective(x, task):
    x = np.clip(x, 0, 1)
    config = ConfigSpace.Configuration(cs, vector=x)
    res = funcs[task].objective_function(config)

    return res["function_value"]


lower = np.zeros(len(cs.get_hyperparameters()))
upper = np.ones(len(cs.get_hyperparameters()))


if args.start_from_bo:
    X = None
    y = None
    indexD = []
    for t in range(len(tasks) - 1):
        r = json.load(open(os.path.join(args.bo_data, "bo_data_%s.json" % tasks[t])))
        if X is None:
            X = r["X"][:args.n_init]
            y = r["y"][:args.n_init]
        else:
            X = np.concatenate((X, r["X"][:args.n_init]), axis=0)
            y = np.concatenate((y, r["y"][:args.n_init]), axis=0)
        indexD.extend([t] * args.n_init)
    indexD = np.array(indexD)

else:
    X = []
    y = []
    indexD = []

    for t in range(len(tasks) - 1):
        for i in range(args.n_init):
            x = cs.sample_configuration().get_array()
            r = objective(x, task=t)

            X.append(x)
            y.append(r)
            indexD.append(t)

    X = np.array(X)
    y = np.array(y)
    indexD = np.array(indexD)

Y, Y_mean, Y_std = normalize_Y(y[:, None], indexD, target_task-1)

# run TEBO
Y_best = []
runtime = []
target_func = partial(objective, task=target_task)
bounds_arr = np.concatenate((lower[:, None], upper[:, None]), axis=1)

# initial design

m = initialize_model(X, Y, indexD, Q, Q_h, M_init, nSamples, ctx, target_task-1, sgp_lengthscale=10, optimize=True,
                     tebo_RBF=False)
m.optimize(max_iters=500, step_rate=0.05)

x_init = make_initial_suggestion(X, Y, indexD, m, Q, Q_h, 100, bounds_arr)[0]
X, Y, indexD, Y_mean, Y_std = evaluate_augment_dataset(target_func, x_init, target_task, X, Y, indexD, Y_mean, Y_std)
Y_best.append(np.min(Y[indexD == target_task]) * Y_std[target_task] + Y_mean[target_task])

# estimate evaluation time
config = ConfigSpace.Configuration(cs, vector=np.clip(x_init, 0, 1))
runtime.append(funcs[-1].objective_function(config)["cost"])

# reinitialize model
m = initialize_model(X, Y, indexD, Q, Q_h, M_init, nSamples, ctx, target_task, sgp_lengthscale=10, optimize=True, tebo_RBF=False)
m.optimize(max_iters=500, step_rate=0.05)

for iter_i in range(1, args.n_iters):
    print("iter %d" % iter_i)

    x_choice = make_new_suggestion(X, Y, indexD, m, Q, Q_h, target_task, 100, bounds_arr)[0]

    X, Y, indexD, Y_mean, Y_std = evaluate_augment_dataset(target_func, x_choice, target_task, X, Y, indexD, Y_mean,
                                                           Y_std)
    m = initialize_model_from_existing(m, X, Y, indexD, Q, Q_h, nSamples, ctx, target_task, tebo_RBF=False)
    m.optimize(max_iters=200, step_rate=0.05)

    # save_model(X, Y, indexD, m, ('./'+exp_name+'_%d')%iter_no, Y_mean, Y_std)

    Y_best.append(np.min(Y[indexD == target_task]) * Y_std[target_task] + Y_mean[target_task])

    # estimate evaluation time
    config = ConfigSpace.Configuration(cs, vector=np.clip(x_choice, 0, 1))
    runtime.append(funcs[-1].objective_function(config)["cost"])


# offline evaluation
results = dict()
results["error"] = [float(yi) for yi in Y_best]
results["benchmark"] = args.main_task
results["run_id"] = args.run_id
results["runtime"] = runtime

if args.start_from_bo:
    fh = open(os.path.join(args.output_path, "tebo_" + args.main_task + '_run_%d_bo_init.json' % args.run_id), 'w')
else:
    fh = open(os.path.join(args.output_path, "tebo_" + args.main_task + '_run_%d.json' % args.run_id), 'w')
json.dump(results, fh)
fh.close()
