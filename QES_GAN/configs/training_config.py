import os
import math
from datetime import datetime

# Utils parameters
ARCHITECTURE_NAME = "QES-WGAN-development"
current_time = datetime.now()
STRING_TIME = current_time.strftime("%Y-%m-%d-%H%M")
OUTPUT_DIR = os.path.join(f"./output/{ARCHITECTURE_NAME}/{STRING_TIME}")

#####################
## GAN PARAMETERS ##
#####################
BATCH_SIZE = 128  # keep high cause distance is calculated on only 1 batch
LR_G = 0.01  # learning rate for the generator
# Betas, initial decay rate for the Adam optimizer
# Check: if these values are appropriate
B1 = 0    # Beta1, the exponential decay rate for the 1st moment estimates. Default would be 0.9
B2 = 0.9  # Beta2, the exponential decay rate for the 2nd moment estimates. Default would be 0.999
LAMBDA_GP = 10  # Coefficient for the gradient penalty

######################
## IMAGE PARAMETERS ##
######################

IMAGE_SIDE = 28
CLASSES = [0, 1]  # This only works for MNIST, picks number classes as specified in list
# Note: for development phase, I assume square images
N_PIXELS = IMAGE_SIDE ** 2

########################
## CIRCUIT PARAMETERS ##
########################
# Note: assuming patches are rows
# N_PATCHES = IMAGE_SIDE
# Note: if instead each image is generated at once:
N_PATCHES = 1
PIXELS_PER_PATCH = int(N_PIXELS / N_PATCHES)
PATCH_WIDTH = int(IMAGE_SIDE)  # Note: executive decision that each patch is a row or multiple few
# rows
PATCH_HEIGHT = int((N_PIXELS / N_PATCHES) / PATCH_WIDTH)
N_ANCILLAS = 1
# Data qubit determined by the number of qubits required to generate as many pixels as needed per
# one patch
N_DATA_QUBITS = math.ceil(math.log(int((IMAGE_SIDE * IMAGE_SIDE) / N_PATCHES), 2))

#############################
## EVOLUTIONARY PARAMETERS ##
#############################

# FITNESS_FUNCTION = 'critic'
FITNESS_FUNCTION = 'emd'
N_CHILDREN = 16
M_MAX_EVALUATIONS = 800
SHOTS = 5000  # maybe 10000-20000 better for 11 qubits
SIMULATOR = 'statevector'
NOISE = False
DTHETA = 0.1
ACTION_WEIGHTS = [60, 10, 10, 20]
MULTI_ACTION_PB = 0.1
MAX_GEN_NO_IMPROVEMENT = 10
MAX_DEPTH = 20
