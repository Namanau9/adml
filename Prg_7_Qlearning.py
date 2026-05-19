import numpy as np
import matplotlib.pyplot as plt


class Gridworld:

    def __init__(self):

        self.size = (3, 3)
        self.goal = (2, 2)
        self.state = (0, 0)

    def reset(self):

        self.state = (0, 0)

        return self.state

    def step(self, action):

        actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        next_state = (
            self.state[0] + actions[action][0],
            self.state[1] + actions[action][1]
        )

        if 0 <= next_state[0] < 3 and 0 <= next_state[1] < 3:

            self.state = next_state

        return self.state, 0 if self.state == self.goal else -1


env = Gridworld()

q_table = np.zeros((3, 3, 4))

alpha, gamma, epsilon = 0.1, 0.9, 0.1

episodes = 1000

for _ in range(episodes):

    state = env.reset()

    done = False

    while not done:

        if np.random.rand() < epsilon:

            action = np.random.choice(4)

        else:

            action = np.argmax(q_table[state[0], state[1]])

        next_state, reward = env.step(action)

        q_table[state[0], state[1], action] = (
            reward + gamma * np.max(q_table[next_state[0], next_state[1]])
        )

        state = next_state

        if state == env.goal:

            done = True


fig, ax = plt.subplots(figsize=(5, 5))

ax.set_ylim(3, 0)

ax.set_xlim(0, 3)

actions = ['↑', '→', '↓', '←']

for y in range(3):

    for x in range(3):

        action = np.argmax(q_table[x, y])

        ax.text(
            x + 0.5,
            y + 0.5,
            actions[action],
            ha='center',
            va='center',
            fontsize=12
        )

ax.text(
    2.5,
    2.5,
    "G",
    ha='center',
    va='center',
    fontsize=12,
    color="green"
)

plt.title("Q-learning Optimal Policy")

plt.show()
