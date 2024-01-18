import random
import matplotlib.pyplot as plt

counter_values = []
prob = 0.91
iterations = 1000


for _ in range(iterations):
    rand = random.uniform(0, 1)
    counter = 0
    while rand < prob:
        counter += 1
        rand = random.uniform(0, 1)
    counter_values.append(counter)

average = sum(counter_values)/len(counter_values)
print(f'Average value: {average}')

plt.hist(counter_values, range=(0, 10))
plt.xlabel('Counter Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of probabilities {prob}')
plt.show()