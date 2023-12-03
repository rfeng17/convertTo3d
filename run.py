import os
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from point_e.util.pc_to_mesh import marching_cubes_mesh
import skimage.measure as measure

from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

# Image folders and  paths:
source_path = "/stable-diffusion/outputs/txt2img-samples/samples"
results_path = "/results"

### Models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)

images_list = []

def extract_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions as needed
            # Load and Preprocess Images
            img_path = os.path.join(folder_path, filename)

            images_list.append(img_path)

def process_images(images, output_folder):
    for image in images:
            img = Image.open(image)
            # Produce a sample from the model.
            samples = None
            for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
                samples = x

            pc = sampler.output_to_point_clouds(samples)[0]

            fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

            plt.show()

            fig_plotly = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=pc.coords[:,0], y=pc.coords[:,1], z=pc.coords[:,2], 
                            mode='markers',
                            marker=dict(
                            size=2,
                            color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(pc.channels["R"], pc.channels["G"], pc.channels["B"])],
                        )
                        )
                    ],
                    layout=dict(
                        scene=dict(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(visible=False)
                        )
                    ),
                )
            
            fig_plotly.show(renderer="colab")

            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            print('creating SDF model...')
            name = 'sdf'
            model = model_from_config(MODEL_CONFIGS[name], device)
            model.eval()

            print('loading SDF model...')
            model.load_state_dict(load_checkpoint(name, device))

            # Produce a mesh (with vertex colors)
            mesh = marching_cubes_mesh(
                pc=pc,
                model=model,
                batch_size=4096,
                grid_size=128, # increase to 128 for resolution used in evals
                progress=True,
            )

            mesh_filename = image + 'mesh.ply'
            mesh_path = os.path.join(output_folder, mesh_filename)

            # Write the mesh to a PLY file to import into some other program.
            with open(mesh_path, 'wb') as f:
                mesh.write_ply(f)

            # Use open3D
            mesh = o3d.io.read_triangle_mesh(mesh_path)

            # Access vertices as a numpy array
            vertices = np.asarray(mesh.vertices)

            # Convert numpy array to Open3D PointCloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(vertices)

            # Estimate normals for the point cloud
            point_cloud.estimate_normals()

            # Poisson surface reconstruction
            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8)

            # Visualize the original mesh
            o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")

            # Remove duplicated vertices
            poisson_mesh.remove_duplicated_vertices()
            # Remove degenerate triangles and unreferenced vertices
            poisson_mesh.remove_degenerate_triangles()
            poisson_mesh.remove_unreferenced_vertices()
            # draw mesh
            poisson_mesh.compute_vertex_normals()
            # Visualize the Poisson reconstructed mesh
            o3d.visualization.draw_geometries([poisson_mesh], window_name="Poisson Reconstruction")

            new_mesh_filename = 'cleaned_' + image + 'mesh.ply'
            mesh_path = os.path.join(output_folder, new_mesh_filename)
            o3d.io.write_triangle_mesh(mesh_path, poisson_mesh)

# Generate a timestamp for the current time
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Create the destination folder with the timestamp
output_folder = os.path.join(results_path, f"result_{timestamp}")

if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

# Copy the source folder to the destination
shutil.copytree(source_path, output_folder)


# Process each image in the folder
extract_images(source_path)
process_images(images_list, output_folder)
