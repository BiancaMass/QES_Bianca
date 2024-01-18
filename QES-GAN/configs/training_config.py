import os
from datetime import datetime

# Utils parameters
ARCHITECTURE_NAME = "QES-WGAN-development"
current_time = datetime.now()
STRING_TIME = current_time.strftime("%Y-%m-%d-%H%M")
OUTPUT_DIR = os.path.join(f"./output/{ARCHITECTURE_NAME}/{STRING_TIME}")

#####################
## GAN PARAMETERS ##
#####################
N_EPOCHS = 2
BATCH_SIZE = 32
LR_G = 0.01  # learning rate for the generator
# Betas, initial decay rate for the Adam optimizer
# Check: if these values are appropriate
B1 = 0    # Beta1, the exponential decay rate for the 1st moment estimates. Default would be 0.9
B2 = 0.9  # Beta2, the exponential decay rate for the 2nd moment estimates. Default would be 0.999
LAMBDA_GP = 10  # Coefficient for the gradient penalty

#####################
## IMAGE VARIABLES ##
#####################

IMAGE_SIZE = 14  # TODO: small for developing purposes. Could increase to 28
CLASSES = [0, 1]  # This only works for MNIST, picks number classes as specified in list



