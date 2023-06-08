# Installation Instructions
## MAM Setup
### Create a conda environment
  
  ```bash
  conda create --name mam python=3.9 -y
  conda activate mam
  ```

### Install packages and other dependencies.

  ```bash
  git clone https://github.com/SHI-Labs/Matting-Anything
  cd Matting-Anything

  # Install all dependencies
  pip install -r requirements.txt

  # Install segment-anything
  python -m pip install -e segment-anything

  # Install Grounding DINO
  export BUILD_WITH_CUDA=True
  export CUDA_HOME=/path/to/cuda/
  python -m pip install -e GroundingDINO

  #Install diffusers
  pip install --upgrade diffusers[torch]
  ```
More details can be found in [segment anything](https://github.com/facebookresearch/segment-anything#installation) and [ GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install) if you meet any installation issues.

### Download the pretrained weights.

  ```bash
  mkdir checkpoints
  cd checkpoints

  # Download GroundingDINO model
  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

  # Download MAM model
  https://drive.google.com/file/d/1GNjlD5A7gN3UZTtzqev5W21w4y21nO8f/view?usp=sharing
  ```

## Gradio Setup
You can setup the gradio demo locally by simply running 
```bash
python gradio_app.py
```
to launch and play with the demo.
We support 3 prompt types in the local Gradio app for MAM：

1. **scribble_point**: Click an point on the target instance for matting.
2. **scribble_box**: Click on two points, the top-left point and the bottom-right point to represent a bounding box of the target instance.
3. **text**: Send text prompt to identify the target instance in the `Text Prompt` box.

We support 2 background types to support image composition with the alpha matte output:

1. **real_world_sample**: Randomly select a real-world image from `assets/backgrounds` for composition.
2. **generated_by_text**: Send background text prompt to create a background image with stable diffusion model in the `Background Prompt` box.

You can alsp play with the demo online at HuggingFace.
