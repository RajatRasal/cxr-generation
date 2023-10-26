python_sources()

pex_binary(
    name="train",
    entry_point="./image_train.py",
    dependencies=[
        "3rdparty:reqs#setuptools",
    ],
    args=["--data_dir", "./.improved_diffusion_training"],
)

pex_binary(
    name="train_linux",
    entry_point="./image_train.py",
    dependencies=[
        "3rdparty:reqs#setuptools",
        "3rdparty:reqs#triton",
        "3rdparty:reqs#nvidia-cublas-cu12",
        "3rdparty:reqs#nvidia-cuda-cupti-cu12",
        "3rdparty:reqs#nvidia-cuda-nvrtc-cu12",
        "3rdparty:reqs#nvidia-cuda-runtime-cu12",
        "3rdparty:reqs#nvidia-cudnn-cu12",
        "3rdparty:reqs#nvidia-cufft-cu12",
        "3rdparty:reqs#nvidia-curand-cu12",
        "3rdparty:reqs#nvidia-cusolver-cu12",
        "3rdparty:reqs#nvidia-cusparse-cu12",
        "3rdparty:reqs#nvidia-nccl-cu12",
        "3rdparty:reqs#nvidia-nvtx-cu12",
    ],
    args=["--data_dir", "./.improved_diffusion_training"],
    platforms=["linux_x86_64-cp-39-cp39m"],
)
