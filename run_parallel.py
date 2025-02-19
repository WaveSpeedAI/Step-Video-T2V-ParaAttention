import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from para_attn.distributed.mp_runner import MPDistRunner

from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed

torch_sdpa = F.scaled_dot_product_attention


class StepVideoPipelineMPDistRunner(MPDistRunner):

    @property
    def world_size(self):
        use_extra_gpu = self.persist_attrs.get("use_extra_gpu", False)

        return torch.cuda.device_count() - (1 if use_extra_gpu else 0)

    def init_processor(self):
        torch.cuda.set_device(dist.get_rank())

        use_quantum_attn = self.persist_attrs.get("use_quantum_attn", False)
        use_fp8_attn = self.persist_attrs.get("use_fp8_attn", False)
        use_fbcache = self.persist_attrs.get("use_fbcache", False)

        if use_quantum_attn:
            from quantum_attn.quantum_attn_interface import attn_func_with_fallback

            F.scaled_dot_product_attention = attn_func_with_fallback

        elif use_fp8_attn:
            from quantum_attn.quantum_attn_interface import fp8_attn_func_with_fallback

            F.scaled_dot_product_attention = fp8_attn_func_with_fallback

        model_dir = self.persist_attrs["model_dir"]
        # vae_url = self.persist_attrs["vae_url"]
        # caption_url = self.persist_attrs["caption_url"]
        use_extra_gpu = self.persist_attrs["use_extra_gpu"]

        self.pipeline = StepVideoPipeline.from_pretrained(model_dir).to(
            dtype=torch.bfloat16).to(device="cuda")
        # self.pipeline.setup_api(
        #     vae_url=vae_url,
        #     caption_url=caption_url,
        # )
        self.pipeline.init_api_models(model_dir, use_extra_gpu=use_extra_gpu)

        from para_attn.context_parallel import init_context_parallel_mesh
        from stepvideo.para_attn.context_parallel import parallelize_pipe

        mesh = init_context_parallel_mesh(self.pipeline.device.type,
                                          max_batch_dim_size=2)
        parallelize_pipe(
            self.pipeline,
            mesh=mesh,
        )

        if use_fbcache:
            from stepvideo.para_attn.first_block_cache import apply_cache_on_pipe

            apply_cache_on_pipe(self.pipeline)

        seed = self.persist_attrs["seed"]
        setup_seed(seed)

        prompt = self.persist_attrs["prompt"]
        num_frames = self.persist_attrs["num_frames"]
        height = self.persist_attrs["height"]
        width = self.persist_attrs["width"]
        cfg_scale = self.persist_attrs["cfg_scale"]
        time_shift = self.persist_attrs["time_shift"]
        pos_magic = self.persist_attrs["pos_magic"]
        neg_magic = self.persist_attrs["neg_magic"]

        begin = time.time()
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        ):
            try:
                self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=1,
                    guidance_scale=cfg_scale,
                    time_shift=time_shift,
                    pos_magic=pos_magic,
                    neg_magic=neg_magic,
                    output_type="latent",
                )
            finally:
                torch.cuda.empty_cache()
        end = time.time()
        print(f"Warmup Time: {end - begin:.2f}s")

    def process_task(self):
        seed = self.persist_attrs["seed"]
        setup_seed(seed)

        prompt = self.persist_attrs["prompt"]
        num_frames = self.persist_attrs["num_frames"]
        height = self.persist_attrs["height"]
        width = self.persist_attrs["width"]
        infer_steps = self.persist_attrs["infer_steps"]
        cfg_scale = self.persist_attrs["cfg_scale"]
        time_shift = self.persist_attrs["time_shift"]
        pos_magic = self.persist_attrs["pos_magic"]
        neg_magic = self.persist_attrs["neg_magic"]

        output_file_name = self.persist_attrs["output_file_name"]

        begin = time.time()
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        ):
            try:
                self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=infer_steps,
                    guidance_scale=cfg_scale,
                    time_shift=time_shift,
                    pos_magic=pos_magic,
                    neg_magic=neg_magic,
                    output_file_name=output_file_name,
                )
            finally:
                torch.cuda.empty_cache()
        end = time.time()
        print(f"Time: {end - begin:.2f}s")


if __name__ == "__main__":
    args = parse_args()

    output_file_name = os.environ.get("OUTPUT_FILE_NAME", "stepvideo")
    use_quantum_attn = os.environ.get("USE_QUANTUM_ATTN") == "1"
    use_fp8_attn = os.environ.get("USE_FP8_ATTN") == "1"
    use_fbcache = os.environ.get("USE_FBCACHE") == "1"
    use_extra_gpu = os.environ.get("USE_EXTRA_GPU") == "1"

    persist_attrs = {
        "model_dir": args.model_dir,
        # "vae_url": args.vae_url,
        # "caption_url": args.caption_url,
        "prompt": args.prompt,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "infer_steps": args.infer_steps,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "time_shift": args.time_shift,
        "pos_magic": args.pos_magic,
        "neg_magic": args.neg_magic,
        "output_file_name": output_file_name,
        "use_quantum_attn": use_quantum_attn,
        "use_fp8_attn": use_fp8_attn,
        "use_fbcache": use_fbcache,
        "use_extra_gpu": use_extra_gpu,
    }

    with StepVideoPipelineMPDistRunner(
            persist_attrs=persist_attrs).start() as runner:
        runner()
