from pytorch_pretrained_bert import BertModel
import torch


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
