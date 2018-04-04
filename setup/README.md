# Installation

A short guide to install this package is below. The package relies on `mujoco-py` which might be the trickiest part of the installation. See `known issues` below and also instructions from the mujoco-py [page](https://github.com/openai/mujoco-py) if you are stuck with mujoco-py installation.

## Linux

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and obtain the mujoco class license key from canvas (talk to instructors to obtain this if you don't have canvas access).
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="path_to_.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
- Install this package using
```
$ conda update conda
$ cd path/to/drl_hw1
$ conda env create -f setup/linux.yml
```
- Add the directory to pythonpath. Change `~/.bashrc` and append:
```
export PYTHONPATH="/path/to/drl_hw1:$PYTHONPATH"
```
- Source the bash so that all paths are correctly configured
```
$ source ~/.bashrc
$ source activate hw1-env
```

## Mac OS

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and obtain the mujoco class license key from canvas (talk to instructors to obtain this if you don't have canvas access).
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Update `~/.bash_profile` by adding the following lines and source it
```
export LD_LIBRARY_PATH="path_to_.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
```
- Install this package using
```
$ conda update conda
$ cd path/to/drl_hw1
$ conda env create -f setup/mac.yml
```
- Add the directory to pythonpath. Change `~/.bash_profile` and append:
```
export PYTHONPATH="/path/to/drl_hw1:$PYTHONPATH"
```
- Source the bash so that all paths are correctly configured
```
$ source ~/.bash_profile
$ source activate hw1-env
```

## Windows

Talk to Kendall

## Known Issues

- Visualization in linux: If the linux system has a GPU, then mujoco-py does not automatically preload the correct drivers. We added an alias `MJPL` in bashrc (see instructions) which stands for mujoco pre-load. When runing any python script that requires rendering, prepend the execution with MJPL.
```
$ MJPL python script.py
```

- Errors related to osmesa during installation. This is a `mujoco-py` build error and would likely go away if the following command is used before creating the conda environment. If the problem still persists, please contact the developers of mujoco-py
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```
