from setuptools import setup, find_packages

setup(
    name="artdiffuse-rl",
    version="0.1.0",
    description="Small latent diffusion model trained from scratch with RL fine-tuning",
    author="ArtDiffuse-RL Contributors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "Pillow>=9.0",
        "numpy>=1.24",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
)
