import torch
import os
from networks.critic import ClassicalCritic
import QES_Bianca as qes
import configs.training_config as training_config


n_data_qubits = training_config.N_DATA_QUBITS
n_ancilla = training_config.N_ANCILLAS
image_shape = (training_config.IMAGE_SIDE, training_config.IMAGE_SIDE)
batch_size = training_config.BATCH_SIZE
n_children = 8
n_max_evaluations = 200
shots = 1000
simulator = 'statevector'
noise = False
dtheta = 0.1
action_weights = [50, 10, 10, 30]
multi_action_pb = 0.1
max_gen_no_improvements = 10
max_depth = 20


# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu = True
    print('GPU available')
else:
    gpu = False

print(f'Using device: {device}')
critic_net = ClassicalCritic(image_shape=(training_config.IMAGE_SIDE, training_config.IMAGE_SIDE))
critic_net = critic_net.to(device)
current_working_directory = os.getcwd()
print("Current Working Directory:", current_working_directory)
critic_net.load_state_dict(torch.load(current_working_directory + '/QES-GAN/output/' + f"critic-640.pt"))  # Note: hardcoded for dev.

qes = qes.Qes(n_data_qubits=n_data_qubits,
              n_ancilla=n_ancilla,
              image_shape=image_shape,
              batch_size=batch_size,
              critic_net=critic_net,
              n_children=n_children,
              n_max_evaluations=n_max_evaluations,
              shots=shots,
              simulator=simulator,
              noise=noise,
              gpu=gpu,
              device=device,
              dtheta=dtheta,
              action_weights=action_weights,
              multi_action_pb=multi_action_pb,
              max_gen_no_improvement=max_gen_no_improvements,
              max_depth=max_depth)

qes.data()
