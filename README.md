## DC-SSDAE: Deep Compression Single-Step Diffusion Autoencoder
The purpose of this project is to prove this architecture can work well among s-o-t-a VAE models, and offers a strong & stable codebase for other VAE researchers to build upon.
### 📜[DC-SSDAE Technical Report](https://mr-for-example.medium.com/dc-ssdae-deep-compression-single-step-diffusion-autoencoder-6d297e5e1a3b)
### 🏋️‍♂️[Experiments Weights & Logs](https://huggingface.co/MrForExample/DC-SSDAE)

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
pip install -r requirements.txt

# download & organize ImageNet-1k dataset
hf auth login
hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /workspace/DC_SSDAE/raw_data
python dc_ssdae/download_imagenet_1k.py --root_dir /workspace/DC_SSDAE
```

### Training
```bash
# If you are training on a remote server, then use `nohup [accelerate cmd below] > test.log 2>&1 &` to prevents processes from being terminated when the terminal session is closed and run the command in the background; 
# You can then use `nvidia-smi`, `ps -ef | grep accelerate` or `ps aux | grep main.py` to check if it's running in the background.
accelerate launch main.py run_name=train_enc_vq_f8c4_FM dataset.im_size=128 dataset.aug_scale=2 training.epochs=200 dc_ssdae.encoder_train=true --config_name=vq_f8c4_FM
accelerate launch main.py run_name=train_enc_dc_f32c32_FM dataset.im_size=128 dataset.aug_scale=2 training.epochs=200 dc_ssdae.encoder_train=true --config_name=dc_f32c32_FM
accelerate launch main.py run_name=train_enc_dc_f32c32_EqM dataset.im_size=128 dataset.aug_scale=2 training.epochs=200 dc_ssdae.encoder_train=true --config_name=dc_f32c32_EqM

# [Optional] Distillation of a model into a single-step decoder
accelerate launch ssdd/main.py run_name=distill_enc_dc_f32c32_EqM training.epochs=10 training.eval_freq=1 dataset.im_size=128 training.lr=1e-4 dc_ssdae.checkpoint=train_enc_dc_f32c32_EqM dc_ssdae.sampler.steps=7 distill_teacher=true
```

### Evaluation
```bash
# Evaluation of multi-steps model
accelerate launch main.py task=eval dataset.im_size=128 dc_ssdae.checkpoint=train_enc_dc_f32c32_EqM dc_ssdae.sampler.steps=8
# [Optional] Evaluation of single-step model
accelerate launch main.py task=eval dataset.im_size=128 dc_ssdae.checkpoint=distill_enc_dc_f32c32_EqM dc_ssdae.sampler.steps=1
```

### References
- [SSDD: Single-Step Diffusion Decoder for Efficient Image Tokenization](https://arxiv.org/abs/2510.04961)
- [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://hanlab.mit.edu/projects/dc-ae)
- [Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models](https://raywang4.github.io/equilibrium_matching)