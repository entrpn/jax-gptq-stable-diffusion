import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from diffusers import FlaxStableDiffusionPipeline
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

import flax

import jax_gptq

tpu = jax.devices('tpu')[0]

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def run(opt):
    if opt.sd_version == 1:
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="bf16",
            dtype=jnp.bfloat16
        )
    else:
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            revision="bf16",
            dtype=jnp.bfloat16
        )
    
    unet = pipeline.unet
    # print('unet params:',unet['params'])

    prompts = ["Labrador in the style of Hokusai"] * opt.batch_size
    prompt_ids = pipeline.prepare_inputs(prompts)
    #prompt_ids = shard(prompt_ids)
    neg_prompt_ids = None

    prompt_embeds = pipeline.text_encoder(prompt_ids, params=params["text_encoder"])[0]
    batch_size = prompt_ids.shape[0]

    max_length = prompt_ids.shape[-1]

    if neg_prompt_ids is None:
        uncond_input = pipeline.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
        ).input_ids
    else:
        uncond_input = neg_prompt_ids
    negative_prompt_embeds = pipeline.text_encoder(uncond_input, params=params["text_encoder"])[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

    latents_shape = (
        opt.batch_size,
        pipeline.unet.in_channels,
            opt.height // pipeline.vae_scale_factor,
            opt.width // pipeline.vae_scale_factor,
    )

    # latents for quantization
    latents = jax.random.normal(create_key(0), shape=latents_shape, dtype=jnp.float32)  

    scheduler_state = pipeline.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=50, shape=latents.shape
        )
    latents = latents * params["scheduler"].init_noise_sigma

    latents_input = jnp.concatenate([latents] * 2)

    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[0]
    timestep = jnp.broadcast_to(t, latents_input.shape[0])
    print(timestep.shape)
    print(timestep.ndim)

    latents_input = pipeline.scheduler.scale_model_input(scheduler_state, latents_input, t)

    context = jnp.array(context,dtype=jnp.bfloat16)

    timestep = jax.device_put(timestep,tpu)
    context = jax.device_put(context, tpu)
    latents_input = jax.device_put(latents_input,tpu)

    def apply_model(params,inp1):
        return unet.apply(
            {"params": params},
            jnp.array(inp1),jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context,
        )

    def apply_model(params,inp1,inp2,inp3):
        return unet.apply(
            {"params": params},
            inp1,inp2,
            encoder_hidden_states=inp3,
        )
    quantized_weights = jax_gptq.quantize(apply_model, params['unet'],jnp.array(latents_input),
                                         jnp.array(timestep, dtype=jnp.int32),
                                         context)
    print(quantized_weights)
    params['unet'] = quantized_weights
    print("done quantizing")
    print(params['unet']) 

    # exit()
    p_params = replicate(params)
    rng = create_key(0)
    rng = jax.random.split(rng, jax.device_count())
    prompts = ["Labrador in the style of Hokusai"] * opt.batch_size

    prompt_ids = pipeline.prepare_inputs(prompts)
    prompt_ids = shard(prompt_ids)

    # Default values https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L275
    num_inference_steps = 50
    height = opt.height 
    width = opt.width 
    guidance_scale = 7.5
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]  # shape: (8, 1)

    # num_inference_steps, height, width, and guidance_scale are static, so need to 
    # specify their positions in the _generate() function as an array to static_broadcasted_argnums
    p_generate = pmap(pipeline._generate, static_broadcasted_argnums=[3,4,5])

    print("Sharded prompt ids has shape:", prompt_ids.shape)
    print("Guidance shape:",g.shape)

    s = time.time()
    images = p_generate(prompt_ids, p_params, rng, num_inference_steps, height, width, g)
    images = images.block_until_ready()
    print("First inference time is:", time.time() - s)

    iters = opt.itters 
    s = time.time()
    for _ in range(iters):
        images = p_generate(prompt_ids, p_params, rng, num_inference_steps, height, width, g)
        images = images.block_until_ready()
    print("Second inference time is:", (time.time() - s)/iters)
    print("Shape of predictions is: ", images.shape)

    if opt.trace:
        trace_path = "/tmp/tensorboard"
        with jax.profiler.trace(trace_path):
            images = p_generate(prompt_ids, p_params, rng, num_inference_steps, height, width, g)
            images = images.block_until_ready()
            print(f"trace can be found: {trace_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Number of images to generate'
    )
    parser.add_argument(
        '--sd-version',
        type=int,
        default=1,
        help='Use 1 for SD 1.4, Use 2 for SD 2.1'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Height'
    )
    parser.add_argument(
        '--itters',
        type=int,
        default=15,
        help='Number of itterations to run the benchmark.'
    )
    parser.add_argument(
        '--trace',
        action="store_true", 
        default=False, 
        help="Run a trace"
    )

    opt = parser.parse_args()
    run(opt)