import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

for episode in range(1000):
    score = 0
    state, _ = env.reset()
    done = False
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Объединяем флаги
        score += reward
        
    print('Episode:', episode, 'Score:', score)
env.close()
