from typing import List, Union, Dict, Optional
import subprocess
import shlex

from simple_slurm import Slurm

def cancel_by_name(job_name: str, signal: bool = False):
    cmd = f"scancel {'--signal=TERM ' if signal else ''}--name {shlex.quote(job_name)}"
    subprocess.run(cmd, shell=True, check=False)
    print(f"Running: {cmd}")
    
def get_slurm(
    machine: str,
    job_type: str,
    job_name: str,
    time: str,
    nice: int,
    partition: str = None,
    nodelist: str = None
):
    slurm_kwargs = {
        "job_name": job_name,
        "output": f"logs/{job_name}%j.out",
        "error": f"logs/{job_name}%j.err",
        # "mail_type": "ALL",
        # "mail_user": "sc3379@cornell.edu",
        "time": time,
    }
    if nice is not None and isinstance(nice, int):
        slurm_kwargs["nice"] = nice

    if machine == "g2":
        slurm_kwargs["nodes"] = 1
        slurm_kwargs["ntasks"] = 1
        slurm_kwargs["get_user_env"] = True
        
        if partition is None or partition == "" or partition == "sun":
            # slurm_kwargs["gres"] = "gpu:a6000:1"
            slurm_kwargs["gres"] = "gpu:nvidia_rtx_a6000:1"
            slurm_kwargs["partition"] = "sun"
            slurm_kwargs["mem"] = "100g"
        elif partition == "thickstun":
            slurm_kwargs["gres"] = "gpu:1"
            slurm_kwargs["partition"] = "thickstun"
            slurm_kwargs["mem"] = "100g"
        elif partition == "desa":
            slurm_kwargs["gres"] = "gpu:nvidia_rtx_a6000:1"
            slurm_kwargs["partition"] = "desa"
            slurm_kwargs["mem"] = "100g"
        else:
            raise ValueError(f"Partition {partition} isn't supported.")
        
        if nodelist is not None and nodelist != "":
            slurm_kwargs["nodelist"] = nodelist

        slurm = Slurm(**slurm_kwargs)

        # Add your environment setup and run command
        slurm.add_cmd('export PATH=/home/sc3379/anaconda3/bin:$PATH')
        slurm.add_cmd('eval "$(conda shell.bash hook)"')
        slurm.add_cmd('conda activate latent_lm')
    elif machine == "gcp":
        if partition is None or partition == "":
            if job_type == "train":
                slurm_kwargs["partition"] = "g2"
            elif job_type == "fid" or job_type == "gen":
                slurm_kwargs["partition"] = "a2"
            else:
                raise ValueError(f"Arguement job_type doesn't accept value {job_type}.")
        else:
            slurm_kwargs["partition"] = partition

        slurm_kwargs["gpus_per_node"] = 1
        slurm = Slurm(**slurm_kwargs)

        # Add your environment setup and run command
        # slurm.add_cmd('export PATH=/home/sc3379/anaconda3/bin:$PATH')
        # slurm.add_cmd('eval "$(conda shell.bash hook)"')
        slurm.add_cmd('source /etc/profile.d/conda.sh')
        slurm.add_cmd('conda activate latent_lm')
    else:
        raise ValueError(f"Arguement machine doesn't accept value {machine}.")
    
    print(f"Slurm configuration for job '{job_name}': {slurm_kwargs}")
    print("---")
    return slurm

def set_job(
    job_name: str,
    machine: str,
    job_type: str,
    time: str = "10:00:00",
    partition: str = None,
    nodelist: str = None,
    nice: int = None,
    dependency: Optional[Union[str, Dict, List]] = None,
) -> Slurm:
    slurm = get_slurm(machine=machine, job_type=job_type, job_name=job_name, time=time, nice=nice, partition=partition, nodelist=nodelist)
    
    if dependency is not None:
        slurm.set_dependency(dependency)
    return slurm