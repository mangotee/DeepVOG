from moviepy.editor import *

name = "torsion_visualization.mp4"
clip = (VideoFileClip(name).subclip((0,1),(0,10)))
clip.write_gif('torsion_visual.gif')

