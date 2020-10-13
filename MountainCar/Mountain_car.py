import gym
import numpy as np

# Creating mountain game env
env = gym.make("MountainCar-v0")
env.reset()


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 10

epsilon = 0.5 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_delay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)



DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# difference between each observation on average
discreate_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n] ))


# creating discreate states
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) /discreate_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
        
    # sending intial state
    discrete_state = get_discrete_state(env.reset())
    print(discrete_state)


    # The game is not done initially
    done = False
    
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        #print(new_state)
        if render:
            env.render()
        if not done:
            # getting action with max reward in for the new state
            max_future_q = np.max(q_table[new_discrete_state])
            # getting q value for current state for that respective action only
            current_q = q_table[discrete_state + (action,)]
            
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # update this q value for current action in q table
            q_table[discrete_state + (action,)] = new_q
        # else setting reward as 0 for final goal
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_delay_value
    
env.close()

