import gym
import numpy as np

def train_gradient_ascent(episodes=30, alpha=0.1):
    env = gym.make("CartPole-v1")
    action_size = env.action_space.n

    baseline = 0
    preferences = np.zeros(action_size)
    probabilities = np.ones(action_size) / action_size

    for e in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.random.choice(action_size, p=probabilities)
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -10
            total_reward += reward

            baseline = baseline + (1 / (steps + 1)) * (reward - baseline)
            for a in range(action_size):
                if a == action:
                    preferences[a] += alpha * (reward - baseline) * (1 - probabilities[a])
                else:
                    preferences[a] -= alpha * (reward - baseline) * probabilities[a]

            exp_preferences = np.exp(preferences)
            probabilities = exp_preferences / np.sum(exp_preferences)
            steps += 1

        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Steps: {steps}")
