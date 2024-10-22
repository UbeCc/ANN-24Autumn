import math
from turtle import position
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

NORM = "pre"
# NORM = "post"
# NORM = "rms"

DEBUG = True

def debug(text):
    if DEBUG:
        print(text)

ACT2FN = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
}

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

class TransposeLinear(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class TfmrAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            # TODO START
            # define the bias term for constructing the causal mask (i.e., seeing only prefix tokens).
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8).view(
                1, 1, max_positions, max_positions                
            ))
            # TODO END
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.c_attn = TransposeLinear(3 * self.embed_dim, self.embed_dim)
        self.c_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value):
        # TODO START
        # implement the multi-head mask self-attnetion mechanism
        # query: (batch_size, num_heads, query_seq_length, head_dim)
        # key: (batch_size, num_heads, key_seq_length, head_dim)
        # value: (batch_size, num_heads, value_seq_length, head_dim)
        # key & value share the same length
   
        # debug(f'q: {query.shape}, k: {key.shape}')
        attn_weights = query @ key.transpose(-1, -2)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        causal_mask = self.bias[..., key.size(-2) - query.size(-2): key.size(-2), :key.size(-2)]
        attn_weights = attn_weights * causal_mask + self.masked_bias.to(attn_weights.device).to(attn_weights.dtype) * (1 - causal_mask)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ value

        return attn_output, attn_weights
        # TODO END

    def _split_heads(self, tensor, num_heads, attn_head_size):
        # TODO START
        # Splits hidden_size dim into attn_head_size and num_heads
        # Input Size: (batch_size, sequence_length, hidden_size)
        # Output Size: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        assert tensor.size(-1) == num_heads * attn_head_size, f"hidden_size: {tensor.size(-1)} != num_heads * attn_head_size: {num_heads * attn_head_size}"
        x_shape = tensor.size()[: -1] + (num_heads, attn_head_size)
        return tensor.view(*x_shape).permute(0, 2, 1, 3)
        # TODO END

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        # TODO START
        # Merges attn_head_size dim and num_attn_heads dim into hidden_size
        # Input Size: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        # Output Size: (batch_size, sequence_length, hidden_size)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        x_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(*x_shape)
        # TODO END

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8).view(
                1, 1, max_positions, max_positions                
            ))
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        config.num_key_value_groups = config.num_attention_heads // 4
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_groups = config.num_key_value_groups
        self.num_kv_heads = self.num_heads // self.num_groups
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.q_proj = TransposeLinear(self.embed_dim, self.embed_dim)
        self.k_proj = TransposeLinear(self.embed_dim // self.num_groups, self.embed_dim)
        self.v_proj = TransposeLinear(self.embed_dim // self.num_groups, self.embed_dim)
        self.o_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value):
        attn_weights = query @ key.transpose(-1, -2)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        causal_mask = self.bias[..., key.size(-2) - query.size(-2): key.size(-2), :key.size(-2)]
        attn_weights = attn_weights * causal_mask + self.masked_bias.to(attn_weights.device).to(attn_weights.dtype) * (1 - causal_mask)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ value

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        assert tensor.size(-1) == num_heads * attn_head_size, f"hidden_size: {tensor.size(-1)} != num_heads * attn_head_size: {num_heads * attn_head_size}"
        x_shape = tensor.size()[: -1] + (num_heads, attn_head_size)
        return tensor.view(*x_shape).permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        x_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(*x_shape)


    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_kv_heads, self.head_dim)
        value = self._split_heads(value, self.num_kv_heads, self.head_dim)

        # Repeat k and v for each group
        key = key.repeat_interleave(self.num_groups, dim=1)
        value = value.repeat_interleave(self.num_groups, dim=1)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class TfmrMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = TransposeLinear(intermediate_size, embed_dim)
        self.c_proj = TransposeLinear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TfmrBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon) if NORM == "rms" else nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # self.attn = TfmrAttention(config)
        self.attn = GQAttention(config)
        self.ln_2 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon) if NORM == "rms" else nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TfmrMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # NOTE: We implement the Pre-Norm version of Transformer, where the ln_1 and ln_2 are place at the residual branch
        # HINT: You can refer to Page 38 in lecture 8 for more details
        hidden_states = attn_output + residual

        if NORM == "pre":
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states
        elif NORM == "post":
            residual = hidden_states
            # residual connection
            hidden_states = self.ln_2(residual + self.mlp(hidden_states))
        elif NORM == "rms":
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states            
        else:
            raise NotImplementedError

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class TfmrModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TfmrBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        inputs_embeds = self.wte(input_ids)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        # TODO START
        # Implement the positional embeddings. Note that the length of cache hidden states used during inference
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.int, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeds = self.wpe(position_ids)
        # TODO END
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions,
        }


class TfmrLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TfmrModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        labels=None,
        use_cache=None,
        PAD_ID=None,
    ):
        # debug(f'PAD_ID: {PAD_ID}')
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            ce_loss_fct = CrossEntropyLoss(reduction="none")
            # TODO START
            # Implement the loss function. Note that you should shift logits so that tokens < n predict n
            # HINT: We set the loss to 0 where [PAD] token is the label, except for the last token, where [PAD] token worked as the "eod of sentence" token.
            
            # Reference: https://discuss.huggingface.co/t/gpt-2-shift-logits-and-labels/812
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            position_mask = torch.ones_like(labels)
            position_mask[:, 1:] = (shift_labels != PAD_ID)
            position_mask = position_mask.float()[:, :-1].contiguous()
            loss = (ce_loss_fct(shift_logits.transpose(1, 2), shift_labels) * position_mask).sum(dim=1) \
                / (position_mask.sum(dim=1) + 1e-5)
            loss = loss.mean()
            
            # padding_mask = shift_labels.eq(PAD_ID)
            # if padding_mask.any():
            #     loss = loss.view(shift_labels.shape)
            #     loss.masked_fill_(padding_mask, 0.0)
            #     loss = loss.sum() / (~padding_mask).sum()
            # else:
            #     loss = loss.mean()
            # TODO END

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "cross_attentions": transformer_outputs["cross_attentions"],
         }
        

    def inference(self, device, PAD_ID, batch_size, maxlen, decode_strategy, temperature, top_p=1.0):
        self.eval()
        allgen = []
        with torch.no_grad():
            for i in range(0, int(5000/batch_size)+1):
                input_ids = torch.tensor([[PAD_ID] for _ in range(batch_size)]).to(device)
                past_key_values = None
                output_ids = input_ids
                for _ in range(maxlen):
                    outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs["logits"]
                    past_key_values = outputs["past_key_values"]
                    logits = logits[:, -1, :] / temperature

                    if decode_strategy == "top-p":
                        # TODO START
                        # implement top-p sampling
                        # shape of logits is (batch_size, num_vocabs)
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumu_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        unused_indices = cumu_probs > top_p
                        unused_indices[:, 1:] = unused_indices[:, :-1].clone() # 
                        unused_indices[:, 0] = 0
                        unused_indices = unused_indices.scatter(1, sorted_indices, unused_indices)
                        logits[unused_indices] = float('-inf')
                        # TODO END
                    prob = logits.softmax(dim=-1) # shape: (batch_size, num_vocabs)
                    now_token = torch.multinomial(prob, 1)[:, :1] # shape: (batch_size)

                    output_ids = torch.cat([output_ids, now_token], 1)
                    input_ids = now_token
                allgen += output_ids.cpu().numpy().tolist()
        pro_allgen = []
        for gen in allgen[:5000]:
            pro_allgen.append([])
            for idx in gen[1:]:
                if idx == PAD_ID:
                    break
                pro_allgen[-1].append(idx)
        self.train() # return to training mode
        return pro_allgen