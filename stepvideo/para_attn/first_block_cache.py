import functools
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import SparseKVAttnMode, UnifiedAttnMode

from stepvideo.modules.model import StepVideoModel


def parallelize_transformer(transformer: StepVideoModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    @functools.wraps(transformer.__class__.prepare_attention_mask)
    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        latent_sequence_length = q_seqlen
        latent_attention_mask = torch.ones(


    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        latent_attention_mask = torch.ones(
            batch_size, 1, latent_sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, 1, N]
        attention_mask = torch.cat(
            [latent_attention_mask, encoder_attention_mask.unsqueeze(1).to(torch.bool)], dim=-1
        )  # [B, 1, N + M]

        world_size = DP.get_world_size(seq_mesh)

        temb = DP.get_assigned_chunk(temb, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)

        hidden_states_len = hidden_states.shape[-2]
        encoder_hidden_states_len = encoder_hidden_states.shape[-2]

        attention_mask = DP.get_assigned_chunk(attention_mask, dim=0, group=batch_mesh)
        attention_mask = attention_mask[..., :1, :]

        new_attention_mask = []
        for i in range(world_size):
            new_attention_mask.append(attention_mask[..., :, i * hidden_states_len : (i + 1) * hidden_states_len])
            new_attention_mask.append(
                attention_mask[
                    ...,
                    :,
                    world_size * hidden_states_len
                    + i * encoder_hidden_states_len : world_size * hidden_states_len
                    + (i + 1) * encoder_hidden_states_len,
                ]
            )
        new_attention_mask = torch.cat(new_attention_mask, dim=-1)
        attention_mask = new_attention_mask

        freqs_cos, freqs_sin = image_rotary_emb

        def get_rotary_emb_chunk(freqs):
            freqs = DP.get_assigned_chunk(freqs, dim=0, group=seq_mesh)
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos)
        freqs_sin = get_rotary_emb_chunk(freqs_sin)
        image_rotary_emb = (freqs_cos, freqs_sin)

        with SparseKVAttnMode(), UnifiedAttnMode(mesh):
            # 4. Transformer blocks
            hidden_states, encoder_hidden_states = self.call_transformer_blocks(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        hidden_states = hidden_states.to(timestep.dtype)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    transformer.forward = new_forward.__get__(transformer)

    def call_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)

        return hidden_states, encoder_hidden_states

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch:
        parallelize_transformer(pipe.transformer, **kwargs)

    return pipe
