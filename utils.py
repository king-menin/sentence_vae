import torch
import numpy as np
from torch.autograd import Variable
from collections import Counter, OrderedDict
import os
import json
import numpy
import bson


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, bson.ObjectId):
            return str(obj)
        else:
            return super(JsonEncoder, self).default(obj)


def jsonify(data):
    return json.dumps(data, cls=JsonEncoder)


def read_config(config):
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            config = json.load(f)
    return config


def save_config(config, path):
    with open(path, "w") as file:
        json.dump(config, file, cls=JsonEncoder)


def if_none(origin, other):
    return other if origin is None else origin


def get_files_path_from_dir(path):
    f = []
    for dir_path, dir_names, filenames in os.walk(path):
        for f_name in filenames:
            f.append(dir_path + "/" + f_name)
    return f


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()] * len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:

            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()

    return sent_str


def interpolate(start, end, steps):
    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps + 2)

    return interpolation.T


def experiment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_" % args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_" % args.embedding_size
    exp_name += "%s_" % args.rnn_type.upper()
    exp_name += "HS=%i_" % args.hidden_size
    exp_name += "L=%i_" % args.num_layers
    exp_name += "BI=%i_" % args.bidirectional
    exp_name += "LS=%i_" % args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_" % args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_" % args.x0
    exp_name += "TS=%s" % ts

    return exp_name
