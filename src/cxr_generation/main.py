import torch
from video_diffusion import Unet3D, GaussianDiffusion

model = Unet3D(
    dim = 64,
    use_bert_text_cond = True,  # this must be set to True to auto-use the bert model dimensions
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,    # height and width of frames
    num_frames = 5,     # number of video frames
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

videos = torch.randn(3, 3, 5, 32, 32) # video (batch, channels, frames, height, width)
 
text = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
]

loss = diffusion(videos, cond = text)
loss.backward()
# after a lot of training

sampled_videos = diffusion.sample(cond = text, cond_scale = 2)
sampled_videos.shape # (3, 3, 5, 32, 32)