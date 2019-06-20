from torch.utils.data import DataLoader
import torch
from rusenttokenize import ru_sent_tokenize
from pytorch_pretrained_bert import BertTokenizer
from utils import read_config, if_none
from tqdm import tqdm
import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer
import re
import os


class InputFeature(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Origin data
            tokens, tok_map):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
        data[2]: list, tokens type ids (for bert)
        """
        self.data = []
        # Bert data
        self.bert_tokens = bert_tokens
        self.input_ids = input_ids
        self.data.append(input_ids)
        self.input_mask = input_mask
        self.data.append(input_mask)
        self.input_type_ids = input_type_ids
        self.data.append(input_type_ids)
        # Origin data
        self.tokens = tokens
        self.tok_map = tok_map

    def __iter__(self):
        return iter(self.data)


class TextDataLoader(DataLoader):
    def __init__(self, data_set, shuffle=False, device="cuda", batch_size=16):
        super(TextDataLoader, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size
        )
        self.device = device

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        for sample in data:
            example = []
            for x in sample:
                example.append(x[:token_ml])
            res.append(example)
        res_ = []
        for x in zip(*res):
            res_.append(torch.LongTensor(x))
        return [t.to(self.device) for t in res_]


class TextDataSet(object):

    @classmethod
    def from_config(cls, config, clear_cache=False, df=None):
        return cls.create(**read_config(config), clear_cache=clear_cache, df=df)

    @classmethod
    def create(cls,
               df_path,
               file_names=None,
               min_char_len=1,
               model_name="bert-base-multilingual-cased",
               max_sequence_length=424,
               pad_idx=0,
               clear_cache=False, df=None):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config = {
            "file_names": file_names,
            "min_char_len": min_char_len,
            "model_name": model_name,
            "max_sequence_length": max_sequence_length,
            "clear_cache": clear_cache,
            "df_path": df_path,
            "pad_idx": pad_idx
        }
        if clear_cache:
            df = cls.files2sentences_df(file_names, min_char_len)
        elif df is None:
            df = pd.read_csv(df_path, sep='\t')
        elif isinstance(df, list):
            df = pd.DataFrame({"text": df})

        self = cls(tokenizer, word_tokenizer=ToktokTokenizer(), df=df, config=config)
        if clear_cache:
            self.save()
        return self

    def load(self, df_path=None):
        df_path = if_none(df_path, self.config["df_path"])
        self.df = pd.read_csv(df_path)

    def create_feature(self, row):
        bert_tokens = []
        orig_tokens = self.word_tokenizer.tokenize(row.text)
        tok_map = []
        for orig_token in orig_tokens:
            cur_tokens = self.tokenizer.tokenize(orig_token)
            if self.config["max_sequence_length"] - 1 < len(bert_tokens) + len(cur_tokens):
                break
            tok_map.append(len(bert_tokens))
            bert_tokens.extend(cur_tokens)

        orig_tokens = ["[CLS]"] + orig_tokens + ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config["max_sequence_length"]:
            input_ids.append(self.config["pad_idx"])
            input_mask.append(0)
            tok_map.append(-1)
        input_type_ids = [0] * len(input_ids)

        return InputFeature(
            # Bert data
            bert_tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            # Origin data
            tokens=orig_tokens,
            tok_map=tok_map,
        )

    def __getitem__(self, item):
        if self.config["df_path"] is None and self.df is None:
            raise ValueError("Should setup df_path or df.")
        if self.df is None:
            self.load()

        return self.create_feature(self.df.iloc[item])

    def __len__(self):
        return len(self.df)

    def save(self, df_path=None):
        df_path = if_none(df_path, self.config["df_path"])
        self.df.to_csv(df_path, sep='\t', index=False)

    @staticmethod
    def files2sentences_df(paths, min_char_len=1):
        # remove tags
        clean = re.compile(r"<.*?>")
        ws_clean = re.compile(r"\s+|\"+")
        res = []
        total = len(paths)
        for f_name in tqdm(paths, total=total, leave=False, desc="files2sentences_df"):
            with open(f_name, "r", encoding="utf-8") as f:
                text = " ".join([x.strip() for x in f.readlines()[2:-1] if len(x.strip())])
                text = clean.sub(" ", text)
                text = ws_clean.sub(" ", text)
                if len(text):
                    for sent in ru_sent_tokenize(text):
                        if len(sent) > min_char_len:
                            res.append(sent)
        return pd.DataFrame({"text": res})

    def __init__(self, tokenizer, word_tokenizer=None, df=None, config=None):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.word_tokenizer = word_tokenizer


class LearnData(object):
    def __init__(self, train_ds=None, train_dl=None, valid_ds=None, valid_dl=None):
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.valid_ds = valid_ds
        self.valid_dl = valid_dl

    @classmethod
    def create(cls,
               # DataSet params
               train_file_names=None,
               train_df_path=None,
               valid_file_names=None,
               valid_df_path=None,
               min_char_len=1,
               model_name="bert-base-multilingual-cased",
               max_sequence_length=424,
               pad_idx=0,
               clear_cache=False,
               # DataLoader params
               device="cuda", batch_size=32):
        train_ds = None
        train_dl = None
        valid_ds = None
        valid_dl = None
        if train_df_path is not None:
            train_ds = TextDataSet.create(
                train_df_path,
                train_file_names,
                min_char_len=min_char_len,
                model_name=model_name,
                max_sequence_length=max_sequence_length,
                pad_idx=pad_idx, clear_cache=clear_cache)
            train_dl = TextDataLoader(train_ds, device=device, shuffle=True, batch_size=batch_size)
        if valid_df_path is not None:
            valid_ds = TextDataSet.create(
                train_df_path,
                valid_file_names,
                min_char_len=min_char_len,
                model_name=model_name,
                max_sequence_length=max_sequence_length,
                pad_idx=pad_idx, clear_cache=clear_cache)
            valid_dl = TextDataLoader(valid_ds, device=device, batch_size=batch_size)

        return cls(train_ds, train_dl, valid_ds, valid_dl)

    def load(self):
        if self.train_ds is not None:
            self.train_ds.load_features()
        if self.valid_ds is not None:
            self.valid_ds.load_features()

    def save(self):
        if self.train_ds is not None:
            self.train_ds.save()
        if self.valid_ds is not None:
            self.valid_ds.save()
