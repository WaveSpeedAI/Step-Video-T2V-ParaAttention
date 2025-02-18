import functools
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import DiffusionPipeline
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

    @functools.wraps(transformer.__class__.prepare_attn_mask)
    def new_prepare_attn_mask(
        self, encoder_attention_mask, encoder_hidden_states, q_seqlen
    ):
        attention_mask = encoder_attention_mask.unsqueeze(1).to(torch.bool)
        return encoder_hidden_states, attention_mask

    transformer.prepare_attn_mask = new_prepare_attn_mask.__get__(transformer)

    @functools.wraps(transformer.__class__.block_forward)
    def new_block_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        rope_positions=None,
        attn_mask=None,
        parallel=True,
    ):
        timestep = DP.get_assigned_chunk(timestep, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(
            encoder_hidden_states, dim=0, group=batch_mesh
        )
        encoder_hidden_states = DP.get_assigned_chunk(
            encoder_hidden_states, dim=-2, group=seq_mesh
        )

        with SparseKVAttnMode(), UnifiedAttnMode(mesh):
            hidden_states = self.call_transformer_blocks(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                attn_mask=attn_mask,
                rope_positions=rope_positions,
            )

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)

        return hidden_states

    transformer.block_forward = new_block_forward.__get__(transformer)

    def call_transformer_blocks(
        self, hidden_states, encoder_hidden_states, *args, **kwargs
    ):
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
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, *args, **kwargs
                )

        return hidden_states

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(
            self,
            *args,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            **kwargs
        ):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    [1],
                    dtype=torch.int64,
                    device=self.device,
                )
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
