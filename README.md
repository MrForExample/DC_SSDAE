## DC-SSDAE: Deep Compression Single-Step Diffusion Autoencoder

### Install
```bash
# setup conda environment
conda create 

# pytorch 2.8.0+cu128
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# On Linux
pip install triton

# download & organize ImageNet-1k dataset
hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /workspace/raw_data
python dc_ssdae/download_imagenet_1k.py
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