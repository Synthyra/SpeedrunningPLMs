import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass
from torch.nn.attention.flex_attention import create_block_mask
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from model.attention import (
    SelfAttention,
    PairedHeadSelfAttention,
    AttentionContext,
    FLASH_ATTN_AVAILABLE,
)
from model.utils import norm, MLP, Linear


@dataclass
class PLMConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int =  8,
        num_hidden_layers: int = 12,
        vocab_size: int = 33,
        expansion_ratio: float = 2.0,
        attention_soft_cap: float = 64.0,
        add_att_soft_cap: bool = True,
        soft_logit_cap: float = 16.0,
        sliding_window_size: int = 2048,
        tie_embeddings: bool = False,
        max_seq_len: int = 1024,
        max_doc_len: int = 2048,
        long_window_every: int = 4,
        paired_head_layers: Optional[List[int]] = None,
        use_flash_attn: bool = True,
        partial_key_offset: bool = True,
        attn_gate_dim: int = 16,
        value_embed_gate_dim: int = 16,
        skip_gate_dim: int = 16,
        smear_gate_dim: int = 16,
        backout_frac: float = 2 / 3,
        unet: bool = False,
        mlm: bool = False,
        token_dropout: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.expansion_ratio = expansion_ratio
        self.soft_logit_cap = soft_logit_cap
        self.attention_soft_cap = attention_soft_cap
        self.add_att_soft_cap = add_att_soft_cap
        self.sliding_window_size = sliding_window_size
        self.tie_embeddings = tie_embeddings
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.long_window_every = long_window_every
        self.paired_head_layers = paired_head_layers
        self.use_flash_attn = use_flash_attn
        self.partial_key_offset = partial_key_offset
        self.attn_gate_dim = attn_gate_dim
        self.value_embed_gate_dim = value_embed_gate_dim
        self.skip_gate_dim = skip_gate_dim
        self.smear_gate_dim = smear_gate_dim
        self.backout_frac = backout_frac
        self.unet = unet
        self.mlm = mlm
        self.token_dropout = token_dropout


@dataclass
class ESMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


class ValueEmbedding(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
        ])

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, tied_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dense(norm(x))
        x = self.act(x)
        if tied_weight is None:
            x = self.decoder(x) + self.bias
        else:
            x = F.linear(x, tied_weight) + self.bias
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


class TransformerBlock(nn.Module):
    def __init__(self, config: PLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        paired_layers = config.paired_head_layers if config.paired_head_layers is not None else []
        if layer_idx in paired_layers:
            self.attn = PairedHeadSelfAttention(config)
            self.is_paired = True
        else:
            self.attn = SelfAttention(config)
            self.is_paired = False
        self.mlp = MLP(config)
        self.unet = config.unet
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
    
    def forward(
        self,
        x: torch.Tensor,
        attention_ctx: AttentionContext,
        vi: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        key_offset: bool = False,
        resid_lambda: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if resid_lambda is not None:
            x = resid_lambda * x
        if self.unet:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(
            x=norm(x),
            attention_ctx=attention_ctx,
            vi=vi,
            key_offset=key_offset,
        )
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        decay = torch.exp(-torch.linspace(0, 1, steps=self.num_layers))
        self.resid_lambdas = nn.Parameter(decay)
        self.backout_layer = max(0, int(self.num_layers * config.backout_frac) - 1)
        self.long_window_every = config.long_window_every
        self.partial_key_offset = config.partial_key_offset

    def forward(
        self,
        x: torch.Tensor,
        attention_ctx_fn,
        window_size_long: int,
        window_size_short: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_backout = None
        for i, layer in enumerate(self.layers):
            use_long = (i % self.long_window_every == 0)
            window_size = window_size_long if use_long else window_size_short
            attention_ctx = attention_ctx_fn(i, layer.is_paired, window_size)
            key_offset = self.partial_key_offset and use_long
            x = layer(
                x=x,
                attention_ctx=attention_ctx,
                key_offset=key_offset,
                resid_lambda=self.resid_lambdas[i],
                **kwargs,
            )
            if i == self.backout_layer:
                x_backout = x
        return x, x_backout
    

class UnetTransformer(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        assert config.num_hidden_layers % 2 == 0
        self.num_encoder_layers = config.num_hidden_layers // 2
        self.num_decoder_layers = config.num_hidden_layers // 2

        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.skip_lambdas = nn.Parameter(-1.5 * torch.ones(self.num_decoder_layers))
        skip_gate_dim = min(config.skip_gate_dim, config.hidden_size)
        self.skip_gates = nn.ModuleList(
            [Linear(skip_gate_dim, 1) for _ in range(self.num_decoder_layers)]
        )

        self.num_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        decay = torch.exp(-torch.linspace(0, 1, steps=self.num_layers))
        self.resid_lambdas = nn.Parameter(decay)
        self.backout_layer = max(0, int(self.num_layers * config.backout_frac) - 1)
        self.long_window_every = config.long_window_every
        self.partial_key_offset = config.partial_key_offset

    def forward(
        self,
        x: torch.Tensor,
        ve: List[torch.Tensor],
        attention_ctx_fn,
        window_size_long: int,
        window_size_short: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = x
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        x_backout = None
        for i in range(self.num_encoder_layers):
            use_long = (i % self.long_window_every == 0)
            window_size = window_size_long if use_long else window_size_short
            attention_ctx = attention_ctx_fn(i, self.layers[i].is_paired, window_size)
            key_offset = self.partial_key_offset and use_long
            x = self.layers[i](
                x=x,
                attention_ctx=attention_ctx,
                vi=ve_enc[i],
                x0=x0,
                key_offset=key_offset,
                resid_lambda=self.resid_lambdas[i],
                **kwargs,
            )
            skip_connections.append(x)
            if i == self.backout_layer:
                x_backout = x
        
        for i in range(self.num_decoder_layers):
            gate_in = x0[..., : self.skip_gates[i].in_features]
            skip_gate = 2 * torch.sigmoid(self.skip_gates[i](gate_in))
            skip_scale = torch.sigmoid(self.skip_lambdas[i]) * skip_gate
            x = x + self.skip_weights[i] * skip_scale * skip_connections.pop()
            layer_idx = self.num_encoder_layers + i
            use_long = (layer_idx % self.long_window_every == 0)
            window_size = window_size_long if use_long else window_size_short
            attention_ctx = attention_ctx_fn(layer_idx, self.layers[layer_idx].is_paired, window_size)
            key_offset = self.partial_key_offset and use_long
            x = self.layers[self.num_encoder_layers + i](
                x=x,
                attention_ctx=attention_ctx,
                vi=ve_dec[i],
                x0=x0,
                key_offset=key_offset,
                resid_lambda=self.resid_lambdas[layer_idx],
                **kwargs,
            )
            if layer_idx == self.backout_layer:
                x_backout = x
        return x, x_backout


class PLM(PreTrainedModel):
    config_class = PLMConfig
    def __init__(self, config: PLMConfig):
        super().__init__(config)
        self.config = config
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.token_dropout = config.token_dropout

        self.vocab_size = config.vocab_size
        self.n_heads = config.num_attention_heads
        self.sliding_window_size = config.sliding_window_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        smear_gate_dim = min(config.smear_gate_dim, config.hidden_size)
        self.smear_gate = Linear(smear_gate_dim, 1)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.tensor(0.5))

        self.unet = config.unet
        if config.unet:
            self.transformer = UnetTransformer(config)
            self.value_embeds = ValueEmbedding(config)
        else:
            self.transformer = Transformer(config)
    
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, config.soft_logit_cap)
        self.tie_embeddings = config.tie_embeddings
        self.split_embed = False
        if self.tie_embeddings:
            self.lm_head.decoder.weight.requires_grad = False

        self.mlm = config.mlm
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def split_tied_embeddings(self):
        if not self.tie_embeddings or self.split_embed:
            return
        with torch.no_grad():
            self.lm_head.decoder.weight.copy_(self.embedding.weight)
        self.lm_head.decoder.weight.requires_grad = True
        self.split_embed = True

    def update_yarn(self, old_window: int, new_window: int):
        for layer in self.transformer.layers:
            layer.attn.apply_yarn(old_window, new_window)

    def _build_doc_info(self, input_ids: torch.Tensor):
        eos_positions = (input_ids == self.eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 0:
            last_eos = eos_positions[-1].item()
            doc_ends = eos_positions
        else:
            last_eos = len(input_ids) - 1
            doc_ends = torch.tensor([last_eos], device=input_ids.device)

        doc_starts = torch.cat(
            [torch.zeros(1, device=input_ids.device, dtype=doc_ends.dtype), doc_ends[:-1] + 1]
        )
        doc_lengths = doc_ends - doc_starts + 1
        cu_seqlens = torch.cat(
            [torch.zeros(1, device=input_ids.device, dtype=torch.int32), doc_lengths.cumsum(0).to(torch.int32)]
        )
        max_seqlen = int(doc_lengths.max().item())
        valid_len = int(doc_ends[-1].item()) + 1
        return last_eos, doc_lengths, cu_seqlens, max_seqlen, valid_len

    def get_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        window_size_long: int,
        window_size_short: int,
    ) -> torch.Tensor:
        docs = (input_ids == self.cls_token_id).cumsum(0)
        last_eos, doc_lengths, cu_seqlens, max_seqlen, valid_len = self._build_doc_info(input_ids)
        paired_doc_lengths = doc_lengths * 2
        paired_cu_seqlens = torch.cat(
            [
                torch.zeros(1, device=input_ids.device, dtype=torch.int32),
                paired_doc_lengths.cumsum(0).to(torch.int32),
            ]
        )
        paired_max_seqlen = int(paired_doc_lengths.max().item())
        paired_valid_len = valid_len * 2
        paired_docs = docs.repeat_interleave(2)

        def attention_ctx_fn(layer_idx: int, is_paired: bool, window_size: int) -> AttentionContext:
            if is_paired:
                docs_used = paired_docs
                cu = paired_cu_seqlens
                max_len = paired_max_seqlen
                valid = paired_valid_len
                window = window_size * 2
                n_heads = self.n_heads // 2
            else:
                docs_used = docs
                cu = cu_seqlens
                max_len = max_seqlen
                valid = valid_len
                window = window_size
                n_heads = self.n_heads

            if self.config.use_flash_attn and FLASH_ATTN_AVAILABLE:
                return AttentionContext(
                    attention_mask=None,
                    cu_seqlens=cu,
                    max_seqlen=max_len,
                    window_size=window,
                    valid_len=valid,
                    use_flash=True,
                    is_paired=is_paired,
                )

            def doc_mask_mod(b, h, q_idx, kv_idx):
                in_window = torch.abs(q_idx - kv_idx) <= window
                doc_mask = docs_used[q_idx] == docs_used[kv_idx]
                pad_mask = (q_idx < valid) & (kv_idx < valid)
                return in_window & doc_mask & pad_mask

            attention_mask = create_block_mask(
                mask_mod=doc_mask_mod,
                B=1,
                H=n_heads,
                Q_LEN=docs_used.numel(),
                KV_LEN=docs_used.numel(),
                device=input_ids.device,
            )
            return AttentionContext(
                attention_mask=attention_mask,
                cu_seqlens=None,
                max_seqlen=max_len,
                window_size=window,
                valid_len=valid,
                use_flash=False,
                is_paired=is_paired,
            )

        x = self.embedding(input_ids)

        if self.token_dropout:
            x = x.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            real_token_count = max(valid_len, 1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum().float() / real_token_count
            x = (x * (1 - mask_ratio_observed)).to(x.dtype)

        if x.size(0) > 1:
            smear_gate_out = self.smear_lambda * torch.sigmoid(
                self.smear_gate(x[1:, : self.smear_gate.in_features])
            )
            x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])

        x = norm(x)
        if self.unet:
            ve = self.value_embeds(input_ids)
            x, x_backout = self.transformer(
                x=x,
                ve=ve,
                attention_ctx_fn=attention_ctx_fn,
                window_size_long=window_size_long,
                window_size_short=window_size_short,
            )
        else:
            x, x_backout = self.transformer(
                x=x,
                attention_ctx_fn=attention_ctx_fn,
                window_size_long=window_size_long,
                window_size_short=window_size_short,
            )
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        return x

    def get_vector_embeddings(self, input_ids: torch.Tensor, window_size_long: int, window_size_short: int) -> torch.Tensor:
        docs = (input_ids == self.cls_token_id).cumsum(0)
        x = self.get_last_hidden_state(input_ids, window_size_long, window_size_short)
        x = x.view(-1, self.config.hidden_size) # (S, hidden_size)
        # At this point, x is shape [S, hidden_size]
        # We want to mean-pool across each document index.
        # Convert docs to 0-based so we can do nice indexing
        num_docs = docs.max().item()
        doc_ids = docs - 1  # Now documents are labeled [0, 1, 2, ...]
        # Mean-pool across tokens belonging to each doc
        doc_embeds = []
        for doc_idx in range(num_docs):
            mask = (doc_ids == doc_idx)
            # Collect all token embeddings for this doc and average
            doc_embeds.append(x[mask].mean(dim=0))
        # Stack into [num_documents, hidden_size]
        return torch.stack(doc_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask_rate: torch.Tensor,
        window_size_long: Optional[int] = None,
        window_size_short: Optional[int] = None,
        ) -> torch.Tensor:
        if window_size_long is None:
            window_size_long = self.sliding_window_size
        if window_size_short is None:
            window_size_short = max(window_size_long // 2, 1)

        last_hidden_state = self.get_last_hidden_state(input_ids, window_size_long, window_size_short)

        tied_weight = self.embedding.weight if self.tie_embeddings and not self.split_embed else None
        lm_logits = self.lm_head(norm(last_hidden_state), tied_weight=tied_weight) # (l, v)

        loss = self.ce(
            lm_logits.view(-1, self.vocab_size),
            labels.view(-1).long()
        )
        #if self.training and not self.mlm:
        #    loss = loss / mask_rate

        return loss


if __name__ == "__main__":
    # py -m model.model
    from torchinfo import summary
    config = PLMConfig(
        hidden_size=768,
        num_attention_heads=6,
        num_hidden_layers=24,
        expansion_ratio=8/3,
        unet=True,
    )
    model = PLM(config).cuda()
    summary(model)

    input_ids = torch.randint(0, 33, (1, 100)).cuda()
    output = model(input_ids)
    print(f"loss: {output.loss}")
    print(f"logits: {output.logits[0].shape}")
    print(f"labels: {output.logits[1].shape}")
    print(f"last_hidden_state: {output.last_hidden_state.shape}")
