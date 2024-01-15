import os

import runhouse as rh


def add_lists(list_a, list_b):
    import numpy as np
    return np.add(np.array(list_a), np.array(list_b))


if __name__ == "__main__":
    gpu = rh.cluster(
        name="rh-cluster",
        ips=["146.169.24.168"],
        ssh_creds={
            "ssh_user": "rrr2417",
            "ssh_private_key": "/homes/rrr2417/.ssh/id_ed25519",
            "password": os.environ["SSH_PASSWORD"],
        },
    )
    # gpu.run(
    #     commands=[
    #         """
    #         # PATH=/vol/biomedic3/rrr2417/.local/bin:/vol/biomedic3/rrr2417/.pyenv/bin:$PATH
    #         # PYENV_ROOT=/vol/biomedic3/rrr2417/.pyenv
    #         # PYENV_VERSION=3.9.18
    #         # PIP_CACHE_DIR=/vol/biomedic3/rrr2417/.cache/pip/
    #         # TRANSFORMERS_CACHE=/data/huggingface_cache/
    #         source /vol/biomedic3/rrr2417/cxr-generation/notebooks/diffedit_diffusers/bin/activate
    #         python3 --version
    #         """
    #     ],
    # )
    gpu.run(
        ["PATH=/vol/biomedic3/rrr2417/cxr-generation/notebooks/diffedit_diffusers/bin python3 --version"],
    )

    # env = rh.env(reqs=["numpy"], env_vars={"PYENV_ROOT": "/vol/biomedic3/rrr2417/.pyenv", "PYENV_VERSION": "3.9.18"})
    # gpu.install_packages(env="trial")
    # cluster_env = env.to(gpu, force_install=False)
    # add_lists_remote = rh.function(fn=add_lists).to(system=gpu, env=env)

    # list_a = [1, 2, 3]
    # list_b = [4, 5, 6]
    # add_lists_remote(list_a, list_b)
