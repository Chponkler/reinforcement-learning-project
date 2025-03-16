import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

def train_q_learning(episodes=10, gamma=0.95, epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.01, batch_size=32, max_steps=200):
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = Sequential([
        Dense(24, input_dim=state_size, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    memory = deque(maxlen=2000)

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        steps = 0

        while not done and steps < max_steps:
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                q_values = model.predict(state)
                action = np.argmax(q_values[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            memory.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1

            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next, d in minibatch:
                    target = r
                    if not d:
                        target = r + gamma * np.amax(model.predict(s_next)[0])
                    target_f = model.predict(s)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {e+1}/{episodes}, Steps: {steps}, Epsilon: {epsilon:.4f}, Done: {done}")
