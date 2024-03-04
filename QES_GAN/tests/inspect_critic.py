import torch
import torch.nn.functional as F

from QuantumEvolutionaryAlgorithms.QES_GAN.networks.CGCC import ClassicalGAN

from QuantumEvolutionaryAlgorithms.QES_GAN.utils.plotting import plot_tensor


def generate_max_activating_image(critic_net, image_shape, num_iterations=40000, lr=0.01):
    # Initialize the input image as random noise
    input_image = torch.randn(1, *image_shape, requires_grad=True)

    for i in range(num_iterations):
        # Compute the output of the critic network
        output = critic_net(input_image)
        if i%5000 == 0:
            print(i)
            print(f'Score at the {i} iteration is {output}')

        # Maximize the output by ascending the gradient
        loss = output
        loss.backward()  # Compute gradients
        input_image.data = input_image.data + lr * input_image.grad.data  # Update input image

        # Clamp the pixel values to the valid range (-1, 1)
        input_image.data = torch.clamp(input_image.data, -1, 1)

        # Reset gradients for next iteration
        input_image.grad.data.zero_()

        # plot_tensor(input_image.detach().squeeze())

    # Return the generated image
    plot_tensor(input_image.detach().squeeze())
    return input_image.detach().squeeze()

# Usage example
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_size = 28
image_shape = (image_size, image_size)

input_folder_path = '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis' \
                        '/scripts/QES-Bianca/QuantumEvolutionaryAlgorithms/QES_GAN/input/'
path_to_critic = input_folder_path + "critic_300_classic.pt"

critic_net = ClassicalGAN.ClassicalCritic(image_shape=image_shape)
critic_net.load_state_dict(torch.load(path_to_critic, map_location=device))

for i in range(5):
    max_activating_image = generate_max_activating_image(critic_net, image_shape)
