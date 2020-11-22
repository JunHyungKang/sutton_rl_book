###################################
# han811                          #
# sutton rl book                  #
# chapter 4 : dynamic programming #
# example 4.1                     #
###################################

import numpy as np

class GRID_WORLD:
    # setting default variables
    def __init__(self):
        self.GRID_SIZE = 4
        self.REWARD = -1
        self.action = ['U','D','L','R']
        self.values = np.zeros((self.GRID_SIZE,self.GRID_SIZE)) # value function
        self.action_prob = 0.25 # random action
        self.states = [] # states
        # setting states except for termination state
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if (i == 0 and j == 0) or (i == self.GRID_SIZE - 1 and j == self.GRID_SIZE - 1):
                    continue
                self.states.append([i,j])

    # return the next state function
    def next_state(self, state, action):
        x, y = state
        if (x == 0 and action == 'U') or (x == self.GRID_SIZE - 1 and action == 'D') or (y == 0 and action == 'L') or (y == self.GRID_SIZE - 1 and action == 'R'):
            return [x,y]
        else:
            if action == 'U':
                return [x-1,y]
            elif action == 'D':
                return [x+1,y]
            elif action == 'R':
                return [x,y+1]
            elif action == 'L':
                return [x,y-1]

    # display the value functions
    def display_values(self):
        print('===============')
        print(self.values) 
        print('===============')

    def compute_value_function(self):
        new_values = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.values = new_values.copy()
        iteration = 1
        while True:
            for (i, j) in self.states:
                value = 0
                for action in self.action:
                    next_i, next_j = self.next_state([i, j], action)
                    value += self.action_prob * (self.REWARD + self.values[next_i, next_j]) # jacobi iteration
                    # value += self.action_prob * (self.REWARD + new_values[next_i, next_j]) # gausss-seidel iteration this is faster than jacobi iter

                new_values[i, j] = value

            if np.sum(np.abs(new_values - self.values)) < 1e-4:
                self.values = new_values.copy()
                break

            self.values = new_values.copy()
            iteration += 1

        return self.values, iteration

        
if __name__ == '__main__':
    world = GRID_WORLD()
    # gridworld is a very simple problem so we can find the optimal policy by only one step of policy improvement after policy evaluation
    _, itr = world.compute_value_function()
    world.display_values()
    print('iter =',itr)

