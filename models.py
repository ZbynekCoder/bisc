from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Baselines
# ----------------------------

class BaselineAdapter(nn.Module):
    """Token-wise MLP + pooling (non-sequential baseline)."""
    def __init__(self, num_tokens: int = 60, d_model: int = 64, mlp_layers: int = 2, pool: str = "mean"):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        layers = []
        for _ in range(mlp_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.pool = pool
        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)      # [B,T,D]
        x = self.mlp(x)                # [B,T,D]
        if self.pool == "mean":
            h = x.mean(dim=1)
        else:
            h = x[:, -1]
        logits = self.head(h)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


class GRUBaseline(nn.Module):
    """Embedding -> GRU -> last hidden -> Linear."""
    def __init__(self, num_tokens: int = 60, d_model: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        _, h = self.gru(x)
        logits = self.head(h[-1])
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


# ----------------------------
# Exact executor (Cayley scan)
# ----------------------------

class A5ExactScan(nn.Module):
    """
    Fixed Cayley table + scan.
    Output logits = log(onehot(final_state)).
    """
    def __init__(self, mul_table, id_id: int, num_tokens: int = 60):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        self.register_buffer("mul", mul_table.long())
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                no_scan: bool = False, shuffle_M: bool = False, reset_each_step: bool = False):
        """
        no_scan: do not update state (stay identity)
        shuffle_M: shuffle the Cayley table rows/cols (break group structure)
        reset_each_step: reset state each step (kill accumulation)
        """
        B, T = input_ids.shape
        device = input_ids.device

        mul = self.mul
        if shuffle_M:
            perm = torch.randperm(self.num_tokens, device=device)
            mul = mul[perm][:, perm]

        s = torch.full((B,), self.id_id, device=device, dtype=torch.long)

        if not no_scan:
            for t in range(T):
                if reset_each_step:
                    s.fill_(self.id_id)
                g = input_ids[:, t]
                s = mul[g, s]  # left multiply

        logits_final = torch.full((B, self.num_tokens), -50.0, device=device)
        logits_final.scatter_(1, s.view(-1, 1), 0.0)  # log(onehot)

        loss = self.loss_fn(logits_final, labels) if labels is not None else None
        return logits_final, loss


# ----------------------------
# Route1: router + exact executor (soft scan)
# ----------------------------

class Route1SoftScan(nn.Module):
    """
    Learnable router: token -> distribution over 60 group elements.
    Fixed executor: Cayley table + scan on distribution (soft).
    Aux supervision: CE(router_logits, token_id) to bootstrap identity routing.
    """
    def __init__(self, mul_table, id_id: int, num_tokens: int = 60,
                 d_model: int = 128, temp: float = 1.0, aux_weight: float = 5.0):
        super().__init__()
        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        self.register_buffer("mul", mul_table.long())
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

        self.embed = nn.Embedding(num_tokens, d_model)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_tokens),
        )

        self.temp = float(temp)
        self.aux_weight = float(aux_weight)
        self._aux_weight_override = None

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                no_scan: bool = False, shuffle_M: bool = False, reset_each_step: bool = False):
        """
        no_scan/shuffle_M/reset_each_step: mechanism ablations for executor.
        """
        B, T = input_ids.shape
        device = input_ids.device

        mul = self.mul
        if shuffle_M:
            perm = torch.randperm(self.num_tokens, device=device)
            mul = mul[perm][:, perm]

        x = input_ids
        h = self.embed(x)                       # [B,T,D]
        logits_g = self.router(h)               # [B,T,60]
        p_g = F.softmax(logits_g / self.temp, dim=-1)  # [B,T,60]

        # state distribution
        s_dist = torch.zeros((B, self.num_tokens), device=device)
        s_dist[:, self.id_id] = 1.0

        if not no_scan:
            # precompute "left-multiply operator" as a permutation matrix over ids for each g
            # next_s[j] = sum_{g,prev} p_g[g] * s_dist[prev] where j = mul[g, prev]
            for t in range(T):
                if reset_each_step:
                    s_dist.zero_()
                    s_dist[:, self.id_id] = 1.0

                pg = p_g[:, t]  # [B,60]
                # compute next_s via scatter-add
                next_s = torch.zeros_like(s_dist)
                # for each g, move mass of s_dist to mul[g, :]
                # vectorized: for each prev_state k, it goes to mul[g,k]
                # We'll do per g loop (60) which is fine.
                for g in range(self.num_tokens):
                    dest = mul[g]  # [60] mapping prev->dest
                    next_s.scatter_add_(1, dest.view(1, -1).expand(B, -1), (pg[:, g].view(B, 1) * s_dist))
                s_dist = next_s

        logits_final = (s_dist.clamp_min(1e-9)).log()

        loss = None
        if labels is not None:
            loss_final = self.loss_fn(logits_final, labels)
            loss_route = F.cross_entropy(logits_g.reshape(-1, self.num_tokens), x.reshape(-1))

            w = self.aux_weight
            if self._aux_weight_override is not None:
                w = float(self._aux_weight_override)
            loss = loss_final + w * loss_route

        return logits_final, loss


# ============================
# Frozen GPT-2 baselines + Teacher-state fusion plugin
# ============================

from typing import Tuple

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None


def _compute_prefix_states(input_ids: torch.Tensor, mul: torch.Tensor, id_id: int,
                           shuffle_state: bool = False, reset_state: bool = False) -> torch.Tensor:
    """
    Teacher state s_t computed by exact executor:
      s_0 = id
      s_t = x_t ∘ s_{t-1}  (left multiply)
    Returns:
      state_ids: LongTensor [B, T]
    """
    B, T = input_ids.shape
    device = input_ids.device
    s = torch.full((B,), int(id_id), device=device, dtype=torch.long)
    states = torch.empty((B, T), device=device, dtype=torch.long)

    for t in range(T):
        if reset_state:
            s.fill_(int(id_id))
        s = mul[input_ids[:, t], s]
        states[:, t] = s

    if shuffle_state and T > 1:
        perm = torch.randperm(T, device=device)
        states = states[:, perm]
    return states


class GPT2FrozenBaseline(nn.Module):
    """
    Baseline:
      (trainable token embedding) -> frozen HF GPT-2 backbone -> last hidden -> Linear(60)
    """
    def __init__(self, num_tokens: int = 60, gpt2_name: str = "openai-community/gpt2",
                 local_files_only: bool = False):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers not installed. pip install transformers")

        self.gpt2 = AutoModel.from_pretrained(
            gpt2_name,
            trust_remote_code=False,
            local_files_only=bool(local_files_only),
        )
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.n_embd = int(self.gpt2.config.n_embd)
        self.num_tokens = int(num_tokens)

        # small trainable front-end
        self.tok_emb = nn.Embedding(self.num_tokens, self.n_embd)

        self.head = nn.Linear(self.n_embd, self.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        attn_mask = torch.ones((B, T), device=input_ids.device, dtype=torch.long)

        x = self.tok_emb(input_ids)  # [B,T,H]
        out = self.gpt2(inputs_embeds=x, attention_mask=attn_mask, use_cache=False, return_dict=True)
        h_last = out.last_hidden_state[:, -1]  # [B,H]
        logits = self.head(h_last)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss


class GPT2FrozenStateFusion(nn.Module):
    """
    Frozen HF GPT-2 backbone + teacher state injection via gated residual fusion at one transformer block.

      h <- h + sigmoid(W_h h + W_s s) ⊙ W_d s

    The teacher state s_t is computed exactly from the Cayley table (mul).
    """
    def __init__(self, mul_table, id_id: int, num_tokens: int = 60,
                 gpt2_name: str = "openai-community/gpt2",
                 inject_layer: int = 8, d_state: int = 128,
                 local_files_only: bool = False):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers not installed. pip install transformers")

        if not torch.is_tensor(mul_table):
            mul_table = torch.tensor(mul_table, dtype=torch.long)
        else:
            mul_table = mul_table.long()
        self.register_buffer("mul", mul_table)  # [60,60]
        self.id_id = int(id_id)
        self.num_tokens = int(num_tokens)

        self.gpt2 = AutoModel.from_pretrained(
            gpt2_name,
            trust_remote_code=False,
            local_files_only=bool(local_files_only),
        )
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.n_embd = int(self.gpt2.config.n_embd)

        # trainable small modules
        self.tok_emb = nn.Embedding(self.num_tokens, self.n_embd)
        self.state_emb = nn.Embedding(self.num_tokens, int(d_state))
        self.state_proj = nn.Linear(int(d_state), self.n_embd)

        self.W_h = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.W_s = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.W_d = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.head = nn.Linear(self.n_embd, self.num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

        # cache for hook
        self._cached_s = None
        self._cached_gate_zero = False

        # --- locate transformer blocks robustly across HF versions ---
        blocks = None

        # Case 1: AutoModel returns GPT2LMHeadModel-like wrapper: model.transformer.h
        if hasattr(self.gpt2, "transformer") and hasattr(self.gpt2.transformer, "h"):
            blocks = self.gpt2.transformer.h

        # Case 2: AutoModel returns GPT2Model directly: model.h
        elif hasattr(self.gpt2, "h"):
            blocks = self.gpt2.h

        # (Optional) extra fallbacks for odd wrappers
        elif hasattr(self.gpt2, "model") and hasattr(self.gpt2.model, "h"):
            blocks = self.gpt2.model.h

        if blocks is None:
            raise ValueError(
                "Cannot locate GPT-2 blocks. Expected .transformer.h or .h on the loaded model. "
                f"Got type={type(self.gpt2)} with attrs={dir(self.gpt2)[:30]}..."
            )

        n_layer = len(blocks)
        self.inject_layer = int(max(0, min(int(inject_layer), n_layer - 1)))

        block = blocks[self.inject_layer]
        block.register_forward_hook(self._fusion_hook)
        print(f"[GPT2FrozenStateFusion] backbone type: {type(self.gpt2)}")
        print(f"[GPT2FrozenStateFusion] inject_layer: {self.inject_layer} / n_layer: {n_layer}")



    def _fusion_hook(self, module, inputs, output):
        """
        output may be tuple(hidden_states, ...) or BaseModelOutput-like.
        We modify hidden_states in-place by returning a new output object/tuple.
        """
        if self._cached_s is None or self._cached_gate_zero:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        s = self._cached_s
        gate = torch.sigmoid(self.W_h(hidden) + self.W_s(s))
        delta = self.W_d(s)
        hidden2 = hidden + gate * delta

        if isinstance(output, tuple):
            return (hidden2,) + output[1:]
        return hidden2

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            shuffle_state: bool = False,
            reset_state: bool = False,
            gate_zero: bool = False,
            state_stride: int = 1,  # NEW: inject/refresh every K steps
    ):
        """
        state_stride (=K):
          - K=1: inject fresh teacher state every step (current behavior).
          - K>1: refresh teacher state every K steps; between refreshes, reuse last injected state.
        """
        B, T = input_ids.shape
        device = input_ids.device
        attn_mask = torch.ones((B, T), device=device, dtype=torch.long)

        # --- 1) compute teacher prefix states (exact) ---
        state_ids = _compute_prefix_states(
            input_ids,
            self.mul,
            self.id_id,
            shuffle_state=shuffle_state,
            reset_state=reset_state,
        )  # [B,T] long

        # --- 2) apply stride: only refresh every K steps, otherwise hold last ---
        K = int(state_stride) if state_stride is not None else 1
        if K < 1:
            K = 1

        if K > 1 and T > 0:
            # hold last refreshed state id
            held = state_ids.clone()
            # For each segment [t0, t0+K-1], copy state at t0 to all positions in segment
            for t0 in range(0, T, K):
                t1 = min(t0 + K, T)
                held[:, t0:t1] = state_ids[:, t0:t0 + 1].expand(B, t1 - t0)
            state_ids = held

        # --- 3) embed+project to GPT-2 hidden size ---
        s = self.state_proj(self.state_emb(state_ids))  # [B,T,H]
        x = self.tok_emb(input_ids)  # [B,T,H]

        # set cache for hook (injection happens at inject_layer)
        self._cached_s = s
        self._cached_gate_zero = bool(gate_zero)

        out = self.gpt2(
            inputs_embeds=x,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        h_last = out.last_hidden_state[:, -1]
        logits = self.head(h_last)

        # clear cache
        self._cached_s = None
        self._cached_gate_zero = False

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return logits, loss

