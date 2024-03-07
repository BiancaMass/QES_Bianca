import torch
import os
from networks.critic import ClassicalCritic
import QES_GAN as qes_gan
import configs.training_config as training_config


def main():
    n_data_qubits = training_config.N_DATA_QUBITS
    n_ancilla = training_config.N_ANCILLAS
    image_shape = (training_config.IMAGE_SIDE, training_config.IMAGE_SIDE)
    batch_size = training_config.BATCH_SIZE
    n_children = training_config.N_CHILDREN
    n_max_evaluations = training_config.M_MAX_EVALUATIONS
    shots = training_config.SHOTS
    simulator = training_config.SIMULATOR
    noise = training_config.NOISE
    dtheta = training_config.DTHETA
    action_weights = training_config.ACTION_WEIGHTS
    multi_action_pb = training_config.MULTI_ACTION_PB
    max_gen_no_improvements = training_config.MAX_GEN_NO_IMPROVEMENT
    max_depth = training_config.MAX_DEPTH

    # device = torch.device("cpu")
    if torch.cuda.is_available():
        gpu = True
        print('GPU available')
    else:
        gpu = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nUsing {device} as a device\n')


    print(f'Using device: {device}')
    critic_net = ClassicalCritic(image_shape=(training_config.IMAGE_SIDE, training_config.IMAGE_SIDE))
    critic_net = critic_net.to(device)
    current_working_directory = os.getcwd()
    critic_net_path = current_working_directory + '/input/' + "critic_300_classic.pt"
    print(f'critic net path: {critic_net_path}')
    # Import pre-trained critic net
    critic_net.load_state_dict(torch.load(critic_net_path, map_location=device))

    qes = qes_gan.Qes(n_data_qubits=n_data_qubits,
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


if __name__ == '__main__':
    main()