import numpy as np

#Value Iteration Function
def value_iteration(S, A, P, R, gamma, k):
    '''
    Inputs
    S: Set of all states - import as list
    A: Set of all actions - import as list
    P: Transition probability P(s'|s,a) - import as a function with inputs s', s and a
    R: Reward function R(s,a) - import as a function with input variables s and a
    gamma: Discount factor
    k: Number of iterations until termination

    Outputs
    pi(S): Approximately optimal policy
    V(S): Value function
    '''
    #Custom function for finding state indexes even if comparing arrays
    def find_index(s_prime, S):
        for i, s in enumerate(S):
            if np.array_equal(s, s_prime):
                return i
        raise ValueError("State not found")

    #Set of value functions with initial value at zero
    V = [0]*len(S)
    pi = []

    n = 0
    while n < k:
        n += 1
        #copy for the new set of value functions we will replace the current with
        V_new = V[:]

        #Iterate through each state possibility
        for s in S:
            Q_values = []
            for a in A:
                #Calculate the state-action Q-value with Bellman's equation
                Qsa = R(s,a) + gamma*sum(P(s_prime, s, a)*V[find_index(s_prime, S)] for s_prime in S)
                Q_values.append(Qsa)
            
            V_new[find_index(s, S)] = max(Q_values)

        #replace previous value with new for the next iteration
        V = V_new

        #terminate after we have reached iteration k
    
    #Find optimal policy after value iteration ran
    for s in S:
        Q_values = []
        for a in A:
            #Calculate the state-action Q-value with Bellman's equation
            Qsa = R(s,a) + gamma* sum(P(s_prime, s, a)*V[find_index(s_prime, S)] for s_prime in S)
            Q_values.append(Qsa)

        best_action = np.argmax(Q_values)
        pi.append(A[best_action])

    return V, pi

#Examples 9.27: https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered27

def get_HealthySickProblem():
    '''
    This function retrieves the (S,A,P,R,gamma) combination for example 9.27 from https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered27
    One may compare the results of this to example 9.31 from https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.SS2.html#Ch9.F16

    This function outputs the S, A, P, R, gamma
    This function DOES NOT output k, required for the number of iterations used for the value iteration, please set this yourself.
    '''
    
    S = ["healthy", "sick"]
    A = ["relax", "party"]

    def P(s_prime, s, a):
        #check function for value iteration for defining the function, this is applied for the example 9.31
        #define every possibility as this is unique for each

        if s == "healthy":
            if a == "relax":
                prob_of_healthy = 0.95
            else: #party
                prob_of_healthy = 0.7
        
        else: #if sick
            if a == "relax":
                prob_of_healthy = 0.5
            else: #party
                prob_of_healthy = 0.1
        
        #now check what output for s' we want to know
        if s_prime == "healthy":
            return prob_of_healthy
        else: #if we want to know probability of sick
            return 1 - prob_of_healthy

    def R(s,a):
        #Perhaps similarly, we define some reward for all combinations

        if s == "healthy":
            if a == "relax":
                reward = 7
            else: #party
                reward = 10
        
        else: #if sick
            if a == "relax":
                reward = 0
            else: #party
                reward = 2

        return reward

    gamma = 0.8

    return S, A, P, R, gamma

def get_3x3GridWorld():
    '''
    Example 9.28 talking about generic gridworld problems
    Example 9.32 describing the problem formulated in this function.

    '''

    #State defining
    S = []

    #This section inputs all possible places an agent could lie in our gridworld.
    empty_matrix = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            s = empty_matrix.copy()
            s[i,j] = 1
            S.append(s)

    #Action
    A = ["left", "right", "up", "down"]

    #Transition Probability
    def P(s_prime, s, a):
        #make a copy of state
        new_state = s.copy()

        #find the location of the current agent
        location_index = np.argmax(new_state)
        x, y = np.unravel_index(location_index, new_state.shape)

        #actions in this example is a guaranteed transition
        if a == "left":
            #This following condition ensures that agent stays within the world
            if y - 1 >= 0:
                new_state[x, y] = 0
                new_state[x, y-1] = 1
        
        elif a == "right":
            #Ensure again that remains in grid world
            if y + 1 <= 2:
                new_state[x, y] = 0
                new_state[x, y+1] = 1
        
        elif a == "up":
            if x - 1 >= 0:
                new_state[x, y] = 0
                new_state[x-1, y] = 1
        
        #Final is to go down
        else:
            if x + 1 <= 2:
                new_state[x, y] = 0
                new_state[x+1, y] = 1

        return 1 if np.array_equal(s_prime, new_state) else 0 

    #Reward Function
    def R(s,a):
        reward_matrix = np.array([
            [0, 0, -0.1],
            [0, 10, -0.1],
            [0, 0, -0.1]
        ])

        reward = np.sum(s*reward_matrix)
        return reward
    
    gamma = 0.9
    return S, A, P, R, gamma



