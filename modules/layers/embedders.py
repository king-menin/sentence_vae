from pytorch_pretrained_bert import BertModel
import torch
from bpemb import BPEmb
import numpy as np


class BertEmbedder(torch.nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased', output_all_encoded_layers=False):
        super(BertEmbedder, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.output_all_encoded_layers = output_all_encoded_layers
        self.model_name = model_name
        self.freeze()

    def forward(self, batch):
        """
        batch has the following structure.
            batch[0]: tokens ids
            batch[1]: tokens mask
            batch[2]: tokens type ids
        """
        with torch.no_grad():
            encoded_layers, _ = self.model(
                input_ids=batch[0],
                token_type_ids=batch[2],
                attention_mask=batch[1],
                output_all_encoded_layers=self.output_all_encoded_layers)
        return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


class LowerCasedBPEEmbedder(torch.nn.Module):
    def __init__(self, lang="ru", dim=300, vs=50001, freeze=False):
        super(LowerCasedBPEEmbedder, self).__init__()
        bpemb = BPEmb(lang=lang, dim=dim, vs=vs-1)
        self.model = self.from_pretrained(np.vstack((bpemb.emb.wv.vectors, np.random.uniform(size=300))), freeze=freeze)

    def forward(self, batch):
        return self.model(batch[0])

    @staticmethod
    def from_pretrained(embeddings, freeze=False):
        assert len(embeddings.shape) == 2, 'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
        embedding.weight = torch.nn.Parameter(torch.tensor(embeddings))
        embedding.weight.requires_grad = not freeze
        return embedding
