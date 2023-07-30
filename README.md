# ML in Computer vision

## Setup

Using Micromamba with guides from [here](https://waylonwalker.com/install-micromamba/).

```
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba ./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc
micromamba create -n <env> python=3.11 -y -c conda-forge
conda activate <env>
pip install mediapipe
```
