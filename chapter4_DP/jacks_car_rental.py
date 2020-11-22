###################################
# han811                          #
# sutton rl book                  #
# chapter 4 : dynamic programming #
# example 4.2                     #
###################################

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
import time
class CAR_RENTAL:
    ############################
    # parameter of the problem #
    ############################
    def __init__(self):
        self.max_cars = 20
        self.max_move_cars = 5

        self.rental_lambda1 = 3
        self.rental_lambda2 = 4
        self.return_lambda1 = 3
        self.return_lambda2 = 2
        
        self.discount_factor = 0.9
        
        self.rental_credit = 10
        self.move_car_cost = 2

        self.policy = np.zeros((self.max_cars+1,self.max_cars+1),dtype=np.int)
        self.states = []
        for i in range(0,self.max_cars+1):
            for j in range(0,self.max_cars+1):
                self.states.append([i,j])
        self.values = np.zeros((self.max_cars+1,self.max_cars+1))
        self.actions = np.arange(-self.max_move_cars,self.max_move_cars+1)

        self.poisson_upper_bound = 11 # the lambda over this value becomes zero

        # fix the return number of cars if this value is false then the policy evaluation become ver very slow
        # when this value is true the value functions are very little different with false
        self.fix_return = True
        self.poisson2 = dict()
        self.poisson3 = dict()
        self.poisson4 = dict()
        for i in range(self.poisson_upper_bound):
            self.poisson2[i] = poisson.pmf(i,2)
        for i in range(self.poisson_upper_bound):
            self.poisson3[i] = poisson.pmf(i,3)
        for i in range(self.poisson_upper_bound):
            self.poisson4[i] = poisson.pmf(i,4)
        # dictionary is much faster than using the scipy poisson function each time
        
    #######################
    # calculate the gains #
    #######################
    def calculate_return(self, state, action):
        # initialize total gain zero
        gain = 0.0
        # calculate moving cost
        gain -= self.move_car_cost * abs(action)
        
        for first_location_rental in range(0, self.poisson_upper_bound):
            for second_location_rental in range(0, self.poisson_upper_bound):
                # calculate number of cars in each location after moving cars
                num_first = int(min(state[0]-action, self.max_cars))
                num_second = int(min(state[1]+action, self.max_cars))
                # calculate rental request
                rental_first = min(num_first, first_location_rental)
                rental_second = min(num_second, second_location_rental)
                # calculate cost
                r = (rental_first + rental_second) * self.rental_credit
                num_first -= rental_first
                num_second -= rental_second
                
                prob = self.poisson3[first_location_rental] * self.poisson4[second_location_rental]
                
                if self.fix_return:
                    first_location_return = self.return_lambda1
                    second_location_return = self.return_lambda2
                    num_first_ = min(num_first+first_location_return, self.max_cars)
                    num_second_ = min(num_second+second_location_return, self.max_cars)
                    prob_ = prob
                    gain += prob_ * (r+self.discount_factor*self.values[num_first_,num_second_])
                else:    
                    for first_location_return in range(0, self.poisson_upper_bound):
                        for second_location_return in range(0, self.poisson_upper_bound):
                            num_first_ = min(num_first+first_location_return, self.max_cars)
                            num_second_ = min(num_second+second_location_return, self.max_cars)
                            prob_ = self.poisson3[first_location_return] * self.possion2[second_location_return] * prob
                            gain += prob_ * (r+self.discount_factor*self.values[num_first_,num_second_])
                # print(rental_first)
                # print(rental_second)
                # print(prob)
                # quit()
        return gain
    
    #########################
    # policy iteration step #
    #########################
    def policy_iteration(self):
        new_value = np.zeros((self.max_cars+1,self.max_cars+1))
        num_policy_improved = 0
        policy_improved = False
        d = 0
        while True:
            if policy_improved == True:
                n = 0
                print('policy improvement count:',num_policy_improved)
                num_policy_improved += 1
                new_policy = np.zeros((self.max_cars+1,self.max_cars+1))
                for first, second in self.states:
                    action_gains = []
                    for action in self.actions:
                        # print('policy improvement count:',n,'until',(self.max_cars+1)*(self.max_cars+1)*len(self.actions),end='\r')
                        if (action>=0 and first>=action) or (action<0 and second>=(-action)):
                            action_gains.append(self.calculate_return([first,second],action))
                        else:
                            action_gains.append(-float('inf'))
                        n+=1
                    
                    idx = np.argmax(action_gains)                    
                    new_policy[first,second] = self.actions[idx]
                # checking policy whethere stable
                change = np.sum(new_policy!=self.policy)
                if change == 0:
                    self.policy = new_policy
                    print('policy doesn\'t change')
                    break
                self.policy = new_policy
                policy_improved = False
            
            # num = 0
            for i, j in self.states:
                # print('calculating',num,' end is num equal 441', end='\r')
                new_value[i,j] = self.calculate_return([i,j],self.policy[i,j])
                # num+=1
                
            # policy evaluation until it converge
            if np.sum(np.abs(new_value-self.values)) < 1e-2:
                print('value function converged!')
                self.values[:] = new_value
                policy_improved = True
                continue
            self.values[:] = new_value
            
    #######################
    # display the results #
    #######################
    def show(self,value,policy):
        if value==True:
            x = []
            y = []
            for i in range(0,self.max_cars+1):
                for j in range(0, self.max_cars+1):
                    x.append(i)
                    y.append(j)
            fig = plt.figure('value')
            ax = fig.add_subplot(111, projection='3d')
            z = []
            for i, j in self.states:
                z.append(self.values[i,j])
            ax.scatter(x,y,z)
            ax.set_xlabel('number of cars in first location')
            ax.set_ylabel('number of cars in second location')
            ax.set_zlabel('expected value of gain')

            plt.show()
        if policy==True:
            x = []
            y = []
            for i in range(0,self.max_cars+1):
                for j in range(0, self.max_cars+1):
                    x.append(i)
                    y.append(j)
            fig = plt.figure('policy')
            ax = fig.add_subplot(111, projection='3d')
            z = []
            for i, j in self.states:
                z.append(self.policy[i,j])
            ax.scatter(x,y,z)
            ax.set_xlabel('number of cars in first location')
            ax.set_ylabel('number of cars in second location')
            ax.set_zlabel('number of car moving in night')
            
            plt.show()

if __name__=='__main__':
    env = CAR_RENTAL()
    env.policy_iteration()
    env.show(value=True,policy=False)
    env.show(value=False,policy=True)