# `value_iteration` Package for Python

This is a value iteration package designed for use on Python, solving sequential decision-making problems that are modelled using a standard Markov decision Process.

# Installation Guide

Use pip to install the package from github.

```
pip install 'git+https://github.com/jimmylin0/value_iteration'
```

# How to use

This package takes as input the defined tuples $(S, A, P, R, \gamma)$ from the associated Markov decision process (MDP). These can be defined as either a function or variable, depending on the need of the user.
- $S$: A set of all possible states,
- $A$: A set of all possible actions,
- $P$: The transition probability $P(s'|s, a)$ - the probability of resulting state at $s'$ given action $a$ taken at state $s$,
- $R$: Reward function $R(s, a)$ - reward given when we take action $a$ at state $s$,
- $\gamma$: Discount factor - how much do we value a reward in the future compared to receiving it now.

After defining the MDP tuples, one needs to specify the number of iterations to run the algorithm for by setting the variable $k$. Then we can implement everything into the function.

## Example
We will define the tuples $(S, A, P, R, \gamma)$ based on the problem from example 9.27 from example 9.27 from https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered27. 

This example describes an agent as either sick or healthy, and give the choice to party or rest. The reward gained is different depending on the choices, and we can apply the value iteration algorithm to this to see which are the optimal actions to take in each state.

$S = {\text{healthy}, \text{sick}}$
$A = {\text{relax}, \text{party}}$

Transition probability $P(s'|s, a)$:
| S        | A       | Probability of  s' = healthy  |
| -------- | ------- | ----------------------------- |
| healthy  | relax   | 0.95                          |
| healthy  | party   | 0.7                           |
| sick     | relax   | 0.5                           |
| sick     | party   | 0.1                           |

Reward $R$:
| S        | A       | Probability of  s' = healthy  |
| -------- | ------- | ----------------------------- |
| healthy  | relax   | 7                             |
| healthy  | party   | 10                            |
| sick     | relax   | 0                             |
| sick     | party   | 2                             |

Here is how one may structure the tuples $(S, A, P, R, \gamma)$:
```
import value_iteration.ValueIteration as vi
import numpy

#Define the tuples
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

#Algorithm Running

k = 100 #number of iterations to run
S, A, P, R, gamma = get_HealthySickProblem() #retrieve parameters
dp, action = vi.value_iteration(S, A, P, R, gamma, k) #Use the value iteration algorithm
print(dp)
print(action)
```
```
[35.714285708110964, 23.809523803349055]
['party', 'relax']
```

This problem is also built into the package along with a different 3x3 gridworld problem. One can retrieve this by using the `get_HealthySickProblem()` function and the `get_3x3Gridworld` function respectively, and use as follows:

```
import value_iteration.ValueIteration as vi
import numpy

k = 100
S, A, P, R, gamma = vi.get_HealthySickProblem()
dp, action = vi.value_iteration(S, A, P, R, gamma, k)
print(dp)
print(action)
```
```
[35.714285708110964, 23.809523803349055]
['party', 'relax']
```

# Contributors

This code was written by Jimmy Lin based on the pseudocode provided in Poole and Mackworth (2010).

# References
Poole, D. L., & Mackworth, A. K. (2010). *Artificial Intelligence: foundations of computational agents*. Cambridge University Press.
