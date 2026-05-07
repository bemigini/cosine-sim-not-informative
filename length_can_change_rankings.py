
"""

An example of how changing the lengths of the 
unembeddings can change the rankings.
Therefore, all pairwise cosine similarities
between unembeddings will not give the possible 
rankings of the model. 

"""


import matplotlib.pyplot as plt
import numpy as np

import model_construction



# Classification problem with 7 labels, y_1, ..., y_7.

rng = np.random.default_rng(0)
num_labels = 7
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:grey']


# Get angles for the unembeddings for the first model, m1.
m1_unemb_angles = np.expand_dims(
    np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4]), 1)

m1_unembs = model_construction.get_2dvectors_from_rad_and_length(m1_unemb_angles, 5)


fig, ax = plt.subplots(1, 1, figsize=(7, 6))
for i in range(num_labels):
    ax.scatter(
        m1_unembs.squeeze()[i, 0], m1_unembs.squeeze()[i, 1],
        s=200, c = colours[i], label = f'{i+1}')
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.legend()
fig.show()


# Inputs to the classifier
inputs = np.expand_dims(rng.uniform(-6.5, 6.5, size = (2000, 2)), axis=2)

all_dots = []

for i in range(num_labels):
    current_dots = np.matmul(m1_unembs[i], inputs).squeeze()
    all_dots.append(current_dots)

all_dots = np.array(all_dots)

arg_max_label = np.argmax(all_dots, axis=0)


fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))
for i in range(num_labels):
    label_filter = arg_max_label == i
    inputs_with_label = inputs[label_filter]
    ax.scatter(
        inputs_with_label.squeeze()[:, 0], inputs_with_label.squeeze()[:, 1],
        s=20, alpha=0.5, c = colours[i], linewidth=0.0)
    ax.scatter(
        m1_unembs.squeeze()[i, 0], m1_unembs.squeeze()[i, 1],
        s=200, c = colours[i], label = f'{i+1}')
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.legend(loc = 'upper right')
ax.set_title('model 1', fontsize = 20)
fig.show()


# Second model, m2, has same angles for the unembeddings as m1,
# but the unembedding for label 3 is much shorter, thus removing the
# rankings where label 3 is the most likely.

m2_unembs = m1_unembs.copy()
m2_unembs[2] = np.array([[0, 2]])

all_dots_m2 = []

for i in range(num_labels):
    current_dots = np.matmul(m2_unembs[i], inputs).squeeze()
    all_dots_m2.append(current_dots)

all_dots_m2 = np.array(all_dots_m2)

arg_max_label_m2 = np.argmax(all_dots_m2, axis=0)


fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))
for i in range(num_labels):
    label_filter = arg_max_label_m2 == i
    inputs_with_label = inputs[label_filter]
    ax.scatter(
        inputs_with_label.squeeze()[:, 0], inputs_with_label.squeeze()[:, 1],
        s=20, alpha=0.5, c = colours[i], linewidth=0.0)
    ax.scatter(
        m2_unembs.squeeze()[i, 0], m2_unembs.squeeze()[i, 1],
        s=200, c = colours[i], label = f'{i+1}')
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.legend(loc = 'upper right')
ax.set_title('model 2', fontsize = 20)
fig.show()
