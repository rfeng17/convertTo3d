import os
import subprocess

# For Windows only

# Command to run in the Anaconda Command Prompt
command_to_run = 'path to\\miniconda3\\Scripts\\activate.bat'
command_activate = "conda activate ldm"
command_folder = 'cd path to\\stable-diffusion'
prompt_file = 'path to\\prompts.txt'

get_prompt = input("Enter prompt (use underscores instead of spaces): ")
with open(prompt_file, "w") as f:
  f.write(get_prompt)
#Change path-to/prompts.txt to prompt file path
command_prompt = "python scripts/txt2img.py --from-file path-to/prompts.txt --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 1"

# Create the command to open the Anaconda Command Prompt and run the desired command
full_command = f"{command_to_run} && {command_activate} && {command_folder} &&{command_prompt}"

# Use subprocess to run the command
subprocess.run(["cmd", "/k", full_command])
