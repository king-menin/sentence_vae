from utils import get_files_path_from_dir
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from modules.data import TextDataSet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="ru_data/wiki/")
    parser.add_argument("--train_path", type=str, default="ru_data/wiki/train.csv/")
    parser.add_argument("--valid_path", type=str, default="ru_data/wiki/valid.csv/")
    parser.add_argument("--test_size", type=float, default=0.001)
    parser.add_argument("--min_char_len", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_sequence_length", type=int, default=424)
    parser.add_argument("--pad_idx", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(123)
    args = parse_args()
    file_names = get_files_path_from_dir(args.data_dir)
    train_files, test_files = train_test_split(file_names, test_size=args.test_size)
    train_ds = TextDataSet.create(
        train_files,
        args.train_path,
        min_char_len=args.min_char_len,
        model_name=args.model_name,
        max_sequence_length=args.max_sequence_length,
        pad_idx=args.pad_idx,
        clear_cache=True)
