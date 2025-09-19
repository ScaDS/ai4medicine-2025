# Introduction to Deep Learning

In this session we will introduce the fundamental concepts of deep learning with a focus on medical applications. We will discuss the structure of neural networks, the role of convolution and pooling, the idea of loss functions, and how deep learning can capture complex patterns in pathology images. Key challenges such as data variability, annotation effort, and clinical integration will also be addressed to give context to the use of AI in medicine.

In the hands-on part, using a prepared notebook, participants will train a simple image classifier to distinguish different pathology tissue types, tweak their models performance and compete for the a little suprise. 
In this session, we will use HPC system and services provided at TU Dresden/ZIH. A virtual environment has been already prepared for you, and it will be activated automatically during the hands-on part. 

You can download the slides [here](3.1_dl_introduction.pdf).

## Local environment

If you want to recreate the same environment on your machine, follow the steps below, but keep in mind that the notebooks have been optimized for HPC resources and your laptop most likely will not have enough memory to run them successfully locally without some code tweaks.

1. **Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)** (macOS and Linux, for Windows please refer to the uv link)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies**. Navigate to 'day3.1_dl_introduction' folder and run:
   ```bash
   uv sync
   ```
   This creates a virtual environment at `.venv/` and installs all necessary Python dependencies.

3. **Register the kernel** by running:
   ```bash
   .venv/bin/python -m ipykernel install --user --name workshop_env --display-name="workshop_env"
   ```
   Now the environment should appear in the kernels list when you open any notebook.

