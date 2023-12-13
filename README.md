--version python 3.10

https://github.com/rfeng17/convertTo3d/

# INSTALLATION:

```
pip install -r requirements.txt
pip install plotly -q

git clone https://github.com/openai/point-e
cd point-e
pip install -e . -q
```

## google collab:
https://colab.research.google.com/drive/1f_3eQUUAodyRd3OMnOHyA7bltQrmBsqc?usp=sharing&authuser=1#scrollTo=hvJCVtNX2xhn

## instructions for stable-diffusion (optional):
https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/

<br />

Install conda environment: https://docs.conda.io/projects/miniconda/en/latest/

Open3D tutorial: http://www.open3d.org/docs/0.10.0/tutorial/Advanced/surface_reconstruction.html

## In folder:

* git clone https://github.com/CompVis/stable-diffusion.git

* git clone https://github.com/openai/point-e

# USAGE:
In run.py:

<br />

source_path = "replace with path to /stable-diffusion/outputs/txt2img-samples/samples"

<br />

results_path = "replace with custom path to folder for results"

<br />

* Stable-diffusion is not needed for source_path (use the path to a folder of images instead)
```
python generate.py
```
* You will be asked to input a prompt for image generation through stable-diffusion
```
python run.py
```

* This will take images as input and process them into 3D objects and mesh. Outputs the mesh.ply in the results_folder for further use if necessary
