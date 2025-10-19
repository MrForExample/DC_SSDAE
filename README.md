## DC-SSDAE: Deep Compression Single-Step Diffusion Autoencoder

### Environment Setup
```bash
# On Linux
sudo apt update
sudo apt install gcc g++

# setup conda environment
# install Miniconda: 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-installer.sh
bash miniconda-installer.sh
# open Miniconda prompt
source ~/.bashrc

conda create --name DC_SSDAE python=3.12
conda activate DC_SSDAE
conda install nvidia/label/cuda-12.8.0::cuda-toolkit

# clone this repo
git clone https://github.com/MrForExample/DC_SSDAE.git

# install dependencies
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install triton
pip install -r requirements.txt

# download & organize ImageNet-1k dataset
hf auth login
hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /workspace/DC_SSDAE/raw_data
python dc_ssdae/download_imagenet_1k.py --root_dir /workspace/DC_SSDAE
```

### Training
```bash
accelerate launch dc_ssdae/main.py run_name=train_enc_vq_f8c4_FM dataset.im_size=128 dataset.aug_scale=2 training.lr=1e-4 dc_ssdae.encoder_train=true --config_name=vq_f8c4_FM
accelerate launch dc_ssdae/main.py run_name=train_enc_vq_f8c4_EqM dataset.im_size=128 dataset.aug_scale=2 training.lr=1e-4 dc_ssdae.encoder_train=true --config_name=vq_f8c4_EqM
accelerate launch dc_ssdae/main.py run_name=train_enc_dc_f32c32_FM dataset.im_size=128 dataset.aug_scale=2 training.lr=1e-4 dc_ssdae.encoder_train=true --config_name=dc_f32c32_FM
accelerate launch dc_ssdae/main.py run_name=train_enc_dc_f32c32_EqM dataset.im_size=128 dataset.aug_scale=2 training.lr=1e-4 dc_ssdae.encoder_train=true --config_name=dc_f32c32_EqM
```

### References
- [SSDD: Single-Step Diffusion Decoder for Efficient Image Tokenization](https://arxiv.org/abs/2510.04961)
- [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://hanlab.mit.edu/projects/dc-ae)
- [Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models](https://raywang4.github.io/equilibrium_matching/)