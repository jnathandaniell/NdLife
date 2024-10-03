from life_2d import LifeSpace, LifeViewer
import numpy as np

rng = np.random.default_rng()

big_window = (62, 230)
small_window = (20, 20)
window = small_window
framerate = 20

dim_1 = window[0]
dim_2 = window[1]
space = rng.random(size=(dim_1, dim_2))
space[space >= 0.8] = True
space[space < 0.8] = False

my_life = LifeSpace(space=space, wrap=True)

LifeViewer(life_space=my_life, g=500000000000, framerate=framerate).show()