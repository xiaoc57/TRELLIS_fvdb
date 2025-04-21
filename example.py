import imageio
from PIL import Image
from trellis_fvdb.pipelines import TrellisImageTo3DPipeline
from trellis_fvdb.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
# NOTICE: 
pipeline = TrellisImageTo3DPipeline.from_pretrained("./")
pipeline.cuda()

# Load an image
image = Image.open("assets/example_image/T.png")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")