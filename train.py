import os
import json
import time
import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from collections import defaultdict
from utils import to_var, experiment_name
from modules.models.model import SentenceVAE
from modules.data import LearnData


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    data = LearnData.create(
        train_df_path=args.train_df_path,
        valid_df_path=args.valid_df_path,
        min_char_len=args.min_char_len,
        model_name=args.model_name,
        max_sequence_length=args.max_sequence_length,
        pad_idx=args.pad_idx,
        clear_cache=False,
        # DataLoader params
        device="cuda"
    )

    model = SentenceVAE(
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, experiment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=args.pad_idx)

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for data_loader, split in [(data.train_dl, 'train'), (data.valid_dl, 'valid')]:

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch[0].size(0)

                # for k, v in batch.items():
                #     if torch.is_tensor(v):
                #         batch[k] = to_var(v)
                # print(batch['input'])
                # print()
                # print(batch['length'])
                # Forward pass
                logp, mean, logv, z = model(batch)

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(
                    logp, batch[0],
                    batch[1].sum(-1), mean, logv, args.anneal_function, step, args.k,
                    args.x0)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                # print(tracker['ELBO'], tracker['ELBO'].shape, loss.data.shape)
                # book keepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.view(1, -1).data))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / batch_size,
                                      epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch * len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight, epoch * len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration + 1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                          % (
                          split.upper(), iteration, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
                          KL_loss.item() / batch_size, KL_weight))

            print(
                "%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='ru_data/wiki/')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=1)
    # parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=768)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512)
    parser.add_argument('-nl', '--num_layers', type=int, default=3)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.3)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    parser.add_argument("--train_df_path", type=str, default="ru_data/wiki/train.csv")
    parser.add_argument("--valid_df_path", type=str, default="ru_data/wiki/valid.csv")
    parser.add_argument("--test_size", type=float, default=0.001)
    parser.add_argument("--min_char_len", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_sequence_length", type=int, default=424)
    parser.add_argument("--pad_idx", type=int, default=0)

    args_ = parser.parse_args()

    args_.rnn_type = args_.rnn_type.lower()
    args_.anneal_function = args_.anneal_function.lower()

    assert args_.rnn_type in ['rnn', 'lstm', 'gru']
    assert args_.anneal_function in ['logistic', 'linear']
    assert 0 <= args_.word_dropout <= 1

    main(args_)
