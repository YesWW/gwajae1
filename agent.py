import numpy as np

class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.eps = 1.0
        self.gamma = 0.85
        self.alpha = 0.1
        ###
        self.N = np.zeros((500,6))
        self.G_return = 0
        self.step_list = []
    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment
        
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.rand() < self.eps:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        #return np.random.choice(self.n_actions)
        return action

    def step(self, state, action, reward, next_state, done):
        
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # epsilon decay
            
        if self.mode == 'mc_control':
            self.step_list.append((state, action, reward))
            if done:
                self.G_return = 0
                if self.eps > 0.01:
                    self.eps -= 0.00002              
                for i in range(len(self.step_list)):
                    state, action, reward = self.step_list[i]                    
                    self.G_return = self.G_return * self.gamma + reward
                    self.N[state][action] += 1
                    self.Q[state][action] += (self.G_return - self.Q[state][action]) / self.N[state][action]
                self.step_list = []

        if self.mode == 'q-learning':
            if done:
                if self.eps > 0.01:
                    self.eps -= 0.00002 
            self.Q[state][action] += self.alpha*(reward +self.gamma* max(self.Q[next_state])
                                                  - self.Q[state][action])

                
            
            
        