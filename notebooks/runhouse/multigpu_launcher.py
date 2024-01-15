import argparse

import runhouse as rh
import torch
from nlp_example import training_function

from accelerate.utils import PrepareForLaunch, patch_environment


def launch_train(*args):
    num_processes = torch.cuda.device_count()
    print(f"Device count: {num_processes}")
    with patch_environment(
        world_size=num_processes,
        master_addr="127.0.0.1",
        master_port="29500",
        mixed_precision=args[1].mixed_precision,
    ):
        launcher = PrepareForLaunch(training_function, distributed_type="MULTI_GPU")
        torch.multiprocessing.start_processes(launcher, args=args, nprocs=num_processes, start_method="spawn")


if __name__ == "__main__":
    gpu = rh.cluster(
        name="rh-cluster",
        ips=["ip_addr"],
        ssh_creds={ssh_user:"<username>", ssh_private_key:"<key_path>"},
    )

    # Set up remote function
    reqs = [
        # "pip:./",
        "transformers",
        "xformers",
        "datasets",
        "evaluate",
        "tqdm",
        "scipy",
        "scikit-learn",
        "tensorboard",
        "torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117",
    ]
    launch_train_gpu = rh.function(fn=launch_train, system=gpu, reqs=reqs, name="train_bert_glue")

    # Define train args/config, run train function
    train_args = argparse.Namespace(cpu=False, mixed_precision="fp16")
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    launch_train_gpu(config, train_args, stream_logs=True)

    # launch_train_gpu = rh.function(fn=launch_train, system=gpu, reqs=reqs, name="train_bert_glue")

    # Define train args/config, run train function
    # train_args = argparse.Namespace(cpu=False, mixed_precision="fp16")
    # config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    # launch_train_gpu(config, train_args, stream_logs=True)

    # Alternatively, we can just run as instructed in the README (but only because there's already a wrapper CLI):
	# my_env = rh.env(
    #     name="my_env",
    #     reqs=reqs,  # ["torch", "diffusers", "~/code/my_other_repo"],
    #     setup_cmds=["source activate ~/.bash_rc"],
    #     env_vars={"HOME": "/vol/biomedic3/rrr2417"}
    #     workdir="./",
    # )
    # my_env.to(gpu)
    # gpu.install_packages(reqs)
    # gpu.run(['accelerate launch --multi_gpu accelerate/examples/nlp_example.py'])
