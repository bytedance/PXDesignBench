FROM ai4s-cn-beijing.cr.volces.com/infra/protenix:v0.0.3

# Install Protenix
RUN pip --no-cache-dir install git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd

# Install PXDesignBench dependencies
RUN pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps
RUN pip install posix_ipc einops transformers==4.51.3 optax==0.2.5 dm-haiku==0.0.13
RUN pip install "jax[cuda]==0.4.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install numpy==1.26.3 natsort dm-tree