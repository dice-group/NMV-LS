import torch.nn as nn

from transformer.encoder import Encoder
from transformer.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target, input_lang, output_lang):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source, input_lang)                            # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)