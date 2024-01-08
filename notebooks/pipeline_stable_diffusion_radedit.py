from typing import Any, Callable, Dict, Optional, Union, List

import PIL
import torch
from diffusers.pipelines.stable_diffusion_diffedit.pipeline_stable_diffusion_diffedit import StableDiffusionDiffEditPipeline, StableDiffusionPipelineOutput, preprocess_mask


class StableDiffusionRadEditPipeline(StableDiffusionDiffEditPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_keep: Union[torch.FloatTensor, PIL.Image.Image] = None,
        image_latents: Union[torch.FloatTensor, PIL.Image.Image] = None,
        inpaint_strength: Optional[float] = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_ckip: int = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            mask_image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to mask the generated image. White pixels in the mask are
                repainted, while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, 1, H, W)`.
            mask_keep (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to mask the generated image. White pixels in the mask are
                repainted, while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, 1, H, W)`.
            image_latents (`PIL.Image.Image` or `torch.FloatTensor`):
                Partially noised image latents from the inversion process to be used as inputs for image generation.
            inpaint_strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to inpaint the masked area. Must be between 0 and 1. When `inpaint_strength` is 1, the
                denoising process is run on the masked area for the full number of iterations specified in
                `num_inference_steps`. `image_latents` is used as a reference for the masked area, and adding more
                noise to a region increases `inpaint_strength`. If `inpaint_strength` is 0, no inpainting occurs.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs
        self.check_inputs(
            prompt,
            inpaint_strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if mask_image is None:
            raise ValueError(
                "`mask_image` input cannot be undefined. Use `generate_mask()` to compute `mask_image` from text prompts."
            )
        if mask_keep is None:
            # TODO: Logger - mask being set to 1 - mask_image
            mask_keep = 1 - mask_image
        if image_latents is None:
            raise ValueError(
                "`image_latents` input cannot be undefined. Use `invert()` to compute `image_latents` from input images."
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_ckip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess masks
        mask_image = preprocess_mask(mask_image, batch_size)
        latent_height, latent_width = mask_image.shape[-2:]
        mask_image = torch.cat([mask_image] * num_images_per_prompt)
        mask_image = mask_image.to(device=device, dtype=prompt_embeds.dtype)

        mask_keep = preprocess_mask(mask_keep, batch_size)
        latent_height, latent_width = mask_keep.shape[-2:]
        mask_keep = torch.cat([mask_keep] * num_images_per_prompt)
        mask_keep = mask_keep.to(device=device, dtype=prompt_embeds.dtype)

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, inpaint_strength, device)

        # 6. Preprocess image latents
        if isinstance(image_latents, list) and any(isinstance(l, torch.Tensor) and l.ndim == 5 for l in image_latents):
            image_latents = torch.cat(image_latents).detach()
        elif isinstance(image_latents, torch.Tensor) and image_latents.ndim == 5:
            image_latents = image_latents.detach()
        else:
            image_latents = self.image_processor.preprocess(image_latents).detach()

        latent_shape = (self.vae.config.latent_channels, latent_height, latent_width)
        if image_latents.shape[-3:] != latent_shape:
            raise ValueError(
                f"Each latent image in `image_latents` must have shape {latent_shape}, "
                f"but has shape {image_latents.shape[-3:]}"
            )
        if image_latents.ndim == 4:
            image_latents = image_latents.reshape(batch_size, len(timesteps), *latent_shape)
        if image_latents.shape[:2] != (batch_size, len(timesteps)):
            raise ValueError(
                f"`image_latents` must have batch size {batch_size} with latent images from {len(timesteps)}"
                f" timesteps, but has batch size {image_latents.shape[0]} with latent images from"
                f" {image_latents.shape[1]} timesteps."
            )
        image_latents = image_latents.transpose(0, 1).repeat_interleave(num_images_per_prompt, dim=1)
        image_latents = image_latents.to(device=device, dtype=prompt_embeds.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        latents = image_latents[0].clone()
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # mask the noise from the current timestep with diffedit mask
                noise_pred = mask_image * noise_pred + (1 - mask_image) * noise_pred_uncond

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # mask with inverted latents from current timestep with user provided mask
                latents = mask_keep * image_latents[i] + (1 - mask_keep) * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
