## DC-SSDAE: Deep Compression Single-Step Diffusion Autoencoder

### Install
```bash
# pytorch 2.8.0+cu128
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# On Linux
pip install triton
```

### Training
```bash
accelerate launch dc_ssdae/main.py run_name=train_enc_f32c32 dataset.im_size=128 dataset.aug_scale=2 training.lr=1e-4 dc_ssdae.encoder_train=true
accelerate launch dc_ssdae/main.py run_name=train_enc_f64c32 dataset.im_size=128 dataset.aug_scale=2 training.lr=1e-4 dc_ssdae.encoder_train=true
```

### References
- [SSDD: Single-Step Diffusion Decoder for Efficient Image Tokenization](https://arxiv.org/abs/2510.04961)
- [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://hanlab.mit.edu/projects/dc-ae)
- [Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models](https://raywang4.github.io/equilibrium_matching/)