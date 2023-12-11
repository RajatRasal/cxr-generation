import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer


# unet for imagen
unet = Unet(
    dim=32,
    cond_dim=512,
    dim_mults=(1, 2, 4, 8),
    num_resnet_blocks=3,
    layer_attns=(False, True, True, True),
    layer_cross_attns=(False, True, True, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)
imagen = Imagen(
    unets=(unet),
    image_sizes=(128),
    timesteps=1000,
    cond_drop_prob=0.1,
).cuda()

# mock images (get a lot of this) and text encodings from large T5
text_embeds = torch.randn(64, 256, 768).cuda()
images = torch.randn(64, 3, 256, 256).cuda()

trainer = ImagenTrainer(imagen)

# feed images into imagen, training each unet in the cascade
for x in range(10):
    print(x)
    loss = trainer(
        images,
        text_embeds=text_embeds,
        unet_number=1,            # training on unet number 1 in this example, but you will have to also save checkpoints and then reload and continue training on unet number 2
        max_batch_size=4          # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
    )
    print(loss)
    print()

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm
# images = trainer.sample(texts = [
#     'a puppy looking anxiously at a giant donut on the table',
#     'the milky way galaxy in the style of monet'
# ], cond_scale = 3.)
