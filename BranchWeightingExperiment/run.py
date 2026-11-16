#!/usr/bin/env python3
"""Cross-platform one-command bootstrap: python run.py."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import venv


def main():
    if sys.version_info < (3, 10):
        raise SystemExit('Python 3.10 or newer is required.')
    root=Path(__file__).resolve().parent
    folder=root/'.venv'
    python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():
        print('Creating a private virtual environment...',flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(folder)
        except Exception as error:
            raise SystemExit(f'Could not create a venv: {error}. Install your Python venv/pip components first.')
    requirement=root/'requirements.txt'
    stamp=folder/'.requirements-installed'
    if not stamp.exists() or stamp.read_bytes()!=requirement.read_bytes():
        subprocess.run([str(python),'-m','pip','--disable-pip-version-check','install','-r',str(requirement)],check=True)
        shutil.copyfile(requirement,stamp)
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONNOUSERSITE='1',
               MPLCONFIGDIR=str(root/'.cache'/'matplotlib'))
    return subprocess.call([str(python),str(root/'run_experiment.py'),*sys.argv[1:]],cwd=root,env=env)


if __name__=='__main__':
    raise SystemExit(main())
