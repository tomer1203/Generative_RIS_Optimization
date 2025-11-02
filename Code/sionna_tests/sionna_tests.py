# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

# Other imports
import matplotlib.pyplot as plt
import numpy as np

no_preview = True # Toggle to False to use the preview widget

# Import relevant components from Sionna RT
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies

# Load integrated scene
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile
# scene = load_scene(sionna.rt.scene.etoile) # Try also sionna.rt.scene.etoile


# Only availabe if a preview is open
# if not no_preview:
#     scene.render_to_file(camera="preview",
#                          filename="scene.png",
#                          resolution=[650,500]);
# Create new camera with different configuration
my_cam = Camera(position=[-250,250,150], look_at=[-15,30,28])

# Render scene with new camera*
scene.render(camera=my_cam, resolution=[650, 500], num_samples=512); # Increase num_samples to increase image quality
plt.show()

scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
scene.objects