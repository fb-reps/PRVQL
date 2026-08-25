import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick(kwargs, names, default=None):
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return default


def _matches(value, reference):
    if not isinstance(value, torch.Tensor) or not isinstance(reference, torch.Tensor):
        return False
    return value.dim() == reference.dim() and value.shape[-1] == reference.shape[-1]


def _query_kv(*args, **kwargs):
    q = _pick(kwargs, ("q", "query", "tgt", "target"))
    if q is None and args:
        q = args[0]
    if q is None:
        q = _pick(kwargs, ("src", "features", "memory", "key_value"))
    if q is None:
        return None, None, None, None
    kv = None
    if args:
        candidates = []
        if q is args[0]:
            if len(args) >= 2:
                candidates.append(args[1])
        else:
            if len(args) >= 1:
                candidates.append(args[0])
            if len(args) >= 2:
                candidates.append(args[1])
        for candidate in candidates:
            if _matches(candidate, q):
                kv = candidate
                break
    if kv is None:
        kv = _pick(kwargs, ("kv", "key_value", "memory", "src", "features", "key"), q)
    mask = _pick(kwargs, ("mask", "attn_mask", "src_mask", "tgt_mask", "memory_mask"))
    key_padding_mask = _pick(
        kwargs,
        ("key_padding_mask", "src_key_padding_mask", "tgt_key_padding_mask", "memory_key_padding_mask"),
    )
    return q, kv, mask, key_padding_mask


def _activation(activation):
    if activation in (None, "relu"):
        return F.relu
    return getattr(F, activation, F.relu)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _activation(activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _attention(self, query, key_value, mask, key_padding_mask):
        if mask is not None and isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
            mask = mask.float().masked_fill(mask, float("-inf")).masked_fill(~mask, 0.0)
        return self.self_attn(
            query,
            key_value,
            key_value,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

    def forward(self, *args, **kwargs):
        query, key_value, mask, key_padding_mask = _query_kv(*args, **kwargs)
        if query is None:
            raise ValueError("TransformerEncoderLayer.forward requires a query tensor.")
        if self.norm_first:
            query = query + self.dropout1(self._attention(self.norm1(query), key_value, mask, key_padding_mask))
            query = query + self.dropout2(self.linear2(self.dropout3(self.activation(self.linear1(self.norm2(query))))))
        else:
            query = self.norm1(query + self.dropout1(self._attention(query, key_value, mask, key_padding_mask)))
            query = self.norm2(query + self.dropout2(self.linear2(self.dropout3(self.activation(self.linear1(query))))))
        return query


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self.dropout(x + self.pe[:, : x.size(1)])


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=2,
        num_encoder_layers=None,
        num_decoder_layers=0,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
        max_len=5000,
        **kwargs,
    ):
        super().__init__()
        if num_encoder_layers is None:
            num_encoder_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.batch_first = batch_first
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    batch_first,
                    norm_first,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

    def forward(self, *args, **kwargs):
        query, key_value, mask, key_padding_mask = _query_kv(*args, **kwargs)
        if query is None:
            raise ValueError("Transformer.forward requires a query tensor.")
        if key_value is None:
            key_value = query
        if kwargs.get("use_pos", False):
            query = self.pos(query)
        for layer in self.layers:
            query = layer(query, key_value, mask=mask, key_padding_mask=key_padding_mask)
        return self.norm(query)

    @property
    def encoder(self):
        return self

    @property
    def encoder_layers(self):
        return self.layers


TransformerEncoder = Transformer
TransformerDecoder = Transformer
TransformerDecoderLayer = TransformerEncoderLayer
MultiheadAttention = nn.MultiheadAttention


def build_transformer(config=None, **kwargs):
    if config is not None:
        cfg = getattr(config, "model", config)
        for source, target in (
            ("hidden_dim", "d_model"),
            ("dim", "d_model"),
            ("nhead", "nhead"),
            ("num_heads", "nhead"),
            ("num_transformer", "num_layers"),
            ("num_layers", "num_layers"),
            ("dim_feedforward", "dim_feedforward"),
            ("dropout", "dropout"),
            ("batch_first", "batch_first"),
            ("norm_first", "norm_first"),
        ):
            if target not in kwargs and hasattr(cfg, source):
                kwargs[target] = getattr(cfg, source)
    return Transformer(**kwargs)
