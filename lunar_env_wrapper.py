import gym

class LunarEnvWrapper(gym.Wrapper):
    """
    A Gym wrapper that allows injecting a custom reward function.
    """
    def __init__(self, env, custom_reward_func=None):
        super().__init__(env)
        self.custom_reward_func = custom_reward_func
        self.current_state = None

    def reset(self, **kwargs):
        self.current_state = self.env.reset(**kwargs)
        return self.current_state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        
        if self.custom_reward_func:
            # The custom reward function signature:
            # func(state, action, original_reward, next_state, done)
            reward = self.custom_reward_func(self.current_state, action, reward, next_state, done)
            
        self.current_state = next_state
        return next_state, reward, done, info
