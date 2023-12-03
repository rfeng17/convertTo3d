--version python 3.10

# INSTALLATION:
pip install -r requirements.txt

## instructions for stable-diffusion:
https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/

## google collab:
https://colab.research.google.com/drive/1f_3eQUUAodyRd3OMnOHyA7bltQrmBsqc?usp=sharing&authuser=1#scrollTo=hvJCVtNX2xhn

## In folder:

* git clone https://github.com/CompVis/stable-diffusion.git

* git clone https://github.com/openai/point-e

# USAGE:
In run.py:
<br />
## source_path = "replace with path to /stable-diffusion/outputs/txt2img-samples/samples"
<br />
## results_path = "replace with custom path to folder for results"
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
