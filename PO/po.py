"""
_______________________________________________________________________________________________
 Puma Optimizer Algorithm (POA)  

 Source codes demo version 1.0                                                                      
                                                                                                    
 Developed in Python (converted from MATLAB R2011b(R2021a))                                                                   
 Main Paper: { Cluster Computing
 Authors: Benyamin Abdollahzadeh, Nima Khodadadi, Saeid Barshandeh, Pavel Trojovský ,Farhad Soleimanian Gharehchopogh, El-Sayed M. El-kenawy, Laith Abualigah, Seyedali Mirjalili
 Puma optimizer (PO): a novel metaheuristic optimization algorithm and its application in machine learning
 DOI: 10.1007/s10586-023-04221-5 }                                                                                                                                                            
                                                                                                    
 e-Mail: benyamin.abdolahzade@gmail.com  
_______________________________________________________________________________________________
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
import random

@dataclass
class Solution:
    X: np.ndarray
    Cost: float

def exploration(solutions: List[Solution], lb: Union[float, np.ndarray], ub: Union[float, np.ndarray], 
                dim: int, n_sol: int, cost_function: Callable) -> List[Solution]:
    # Sort solutions by cost
    solutions = sorted(solutions, key=lambda x: x.Cost)
    p_cr = 0.20
    pcr = 1 - p_cr  # Eq 28
    p = pcr / n_sol  # Eq 29
    
    new_solutions = []
    for i in range(n_sol):
        x = solutions[i].X
        # Get random indices excluding current index
        available_indices = list(range(n_sol))
        available_indices.remove(i)
        random_indices = random.sample(available_indices, 6)
        a, b, c, d, e, f = [solutions[idx].X for idx in random_indices]
        
        G = 2 * random.random() - 1  # Eq 26
        
        if random.random() < 0.5:
            y = np.random.uniform(lb, ub, dim)  # Eq 25
        else:
            y = a + G * (a - b) + G * (((a - b) - (c - d)) + ((c - d) - (e - f)))  # Eq 25
            
        y = np.clip(y, lb, ub)
        z = x.copy()
        
        j0 = random.randint(0, dim-1)
        for j in range(dim):
            if j == j0 or random.random() <= p_cr:
                z[j] = y[j]
                
        new_sol = Solution(X=z, Cost=cost_function(z))
        
        if new_sol.Cost < solutions[i].Cost:
            new_solutions.append(new_sol)
        else:
            p_cr += p  # Eq 30
            new_solutions.append(solutions[i])
            
    return new_solutions

def exploitation(solutions: List[Solution], lb: Union[float, np.ndarray], ub: Union[float, np.ndarray],
                dim: int, n_sol: int, best: Solution, max_iter: int, iter: int, 
                cost_function: Callable) -> List[Solution]:
    Q = 0.67
    Beta = 2
    new_solutions = []
    
    for i in range(n_sol):
        beta1 = 2 * random.random()
        beta2 = np.random.randn(dim)
        w = np.random.randn(dim)  # Eq 37
        v = np.random.randn(dim)  # Eq 38
        F1 = np.random.randn(dim) * np.exp(2 - iter * (2/max_iter))  # Eq 35
        F2 = w * v**2 * np.cos((2*random.random())*w)  # Eq 36
        
        # Calculate mean best
        mbest = np.mean([s.X for s in solutions], axis=0) / n_sol
        
        R_1 = 2 * random.random() - 1  # Eq 34
        S1 = 2 * random.random() - 1 + np.random.randn(dim)
        S2 = F1 * R_1 * solutions[i].X + F2 * (1-R_1) * best.X
        VEC = S2 / S1
        
        if random.random() <= 0.5:
            Xatack = VEC
            if random.random() > Q:
                new_x = best.X + beta1 * np.exp(beta2) * (solutions[random.randint(0, n_sol-1)].X - solutions[i].X)  # Eq 32
            else:
                new_x = beta1 * Xatack - best.X  # Eq 32
        else:
            r1 = random.randint(0, n_sol-1)  # Eq 33
            new_x = (mbest * solutions[r1].X - ((-1)**(random.randint(0,1))) * solutions[i].X) / (1 + (Beta * random.random()))  # Eq 32
            
        new_x = np.clip(new_x, lb, ub)
        new_sol = Solution(X=new_x, Cost=cost_function(new_x))
        
        if new_sol.Cost < solutions[i].Cost:
            new_solutions.append(new_sol)
        else:
            new_solutions.append(solutions[i])
            
    return new_solutions

def puma(n_sol: int, max_iter: int, lb: Union[float, np.ndarray], ub: Union[float, np.ndarray], 
         dim: int, cost_function: Callable) -> Tuple[np.ndarray, float, List[float]]:
    """
    Puma Optimizer Algorithm implementation
    
    Parameters:
    -----------
    n_sol : int
        Number of solutions (population size)
    max_iter : int
        Maximum number of iterations
    lb : float or np.ndarray
        Lower bound(s)
    ub : float or np.ndarray
        Upper bound(s)
    dim : int
        Problem dimension
    cost_function : callable
        The objective function to be minimized
        
    Returns:
    --------
    best_x : np.ndarray
        Best solution found
    best_cost : float
        Cost of the best solution
    convergence : list
        Convergence history
    """
    # Parameter setting
    un_selected = np.ones(2)  # 1:Exploration 2:Exploitation
    f3_explore = 0
    f3_exploit = 0
    seq_time_explore = np.ones(3)
    seq_time_exploit = np.ones(3)
    seq_cost_explore = np.ones(3)
    seq_cost_exploit = np.ones(3)
    pf = np.array([0.5, 0.5, 0.3])  # 1&2 for intensification (for F1 and F2) 3 for diversification (For F3)
    pf_f3 = []
    mega_explor = 0.99
    mega_exploit = 0.99
    
    # Initialization
    solutions = []
    for _ in range(n_sol):
        x = np.random.uniform(lb, ub, dim)
        solutions.append(Solution(X=x, Cost=cost_function(x)))
    
    best = min(solutions, key=lambda x: x.Cost)
    initial_best = best
    flag_change = 1
    convergence = []
    
    # Unexperienced Phase
    for iter in range(1, 4):
        solutions = exploration(solutions, lb, ub, dim, n_sol, cost_function)
        costs_explor = min(s.Cost for s in solutions)
        
        solutions = exploitation(solutions, lb, ub, dim, n_sol, best, max_iter, iter, cost_function)
        costs_exploit = min(s.Cost for s in solutions)
        
        best = min(solutions, key=lambda x: x.Cost)
        convergence.append(best.Cost)
        print(f'Iteration: {iter} Best Cost = {best.Cost}')
    
    # Hyper Initialization
    seq_cost_explore[0] = abs(initial_best.Cost - costs_explor)  # Eq 5
    seq_cost_exploit[0] = abs(initial_best.Cost - costs_exploit)  # Eq 8
    seq_cost_explore[1] = abs(costs_explor - seq_cost_explore[0])  # Eq 6
    seq_cost_exploit[1] = abs(costs_exploit - seq_cost_exploit[0])  # Eq 9
    seq_cost_explore[2] = abs(costs_explor - seq_cost_explore[1])  # Eq 7
    seq_cost_exploit[2] = abs(costs_exploit - seq_cost_exploit[1])  # Eq 10
    
    for cost in [*seq_cost_explore, *seq_cost_exploit]:
        if cost != 0:
            pf_f3.append(cost)
    
    # F1_Explore
    f1_explor = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])  # Eq 1
    # F1_Exploit
    f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])  # Eq 2
    # F2_Explore
    f2_explor = pf[1] * (sum(seq_cost_explore) / sum(seq_time_explore))  # Eq 3
    # F2_Exploit
    f2_exploit = pf[1] * (sum(seq_cost_exploit) / sum(seq_time_exploit))  # Eq 4
    
    # Score calculation
    score_explore = (pf[0] * f1_explor) + (pf[1] * f2_explor)  # Eq 11
    score_exploit = (pf[0] * f1_exploit) + (pf[1] * f2_exploit)  # Eq 12
    
    # Experienced Phase
    for iter in range(4, max_iter + 1):
        if score_explore > score_exploit:
            # Exploration
            select_flag = 1
            solutions = exploration(solutions, lb, ub, dim, n_sol, cost_function)
            count_select = un_selected.copy()
            un_selected[1] += 1
            un_selected[0] = 1
            f3_explore = pf[2]
            f3_exploit += pf[2]
            
            temp_best = min(solutions, key=lambda x: x.Cost)
            seq_cost_explore = np.roll(seq_cost_explore, 1)
            seq_cost_explore[0] = abs(best.Cost - temp_best.Cost)
            
            if seq_cost_explore[0] != 0:
                pf_f3.append(seq_cost_explore[0])
                
            if temp_best.Cost < best.Cost:
                best = temp_best
        else:
            # Exploitation
            select_flag = 2
            solutions = exploitation(solutions, lb, ub, dim, n_sol, best, max_iter, iter, cost_function)
            count_select = un_selected.copy()
            un_selected[0] += 1
            un_selected[1] = 1
            f3_explore += pf[2]
            f3_exploit = pf[2]
            
            temp_best = min(solutions, key=lambda x: x.Cost)
            seq_cost_exploit = np.roll(seq_cost_exploit, 1)
            seq_cost_exploit[0] = abs(best.Cost - temp_best.Cost)
            
            if seq_cost_exploit[0] != 0:
                pf_f3.append(seq_cost_exploit[0])
                
            if temp_best.Cost < best.Cost:
                best = temp_best
                
        if flag_change != select_flag:
            flag_change = select_flag
            seq_time_explore = np.roll(seq_time_explore, 1)
            seq_time_explore[0] = count_select[0]
            seq_time_exploit = np.roll(seq_time_exploit, 1)
            seq_time_exploit[0] = count_select[1]
            
        # Hyper Initialization
        f1_explor = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])  # Eq 14
        f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])  # Eq 13
        f2_explor = pf[1] * (sum(seq_cost_explore) / sum(seq_time_explore))  # Eq 16
        f2_exploit = pf[1] * (sum(seq_cost_exploit) / sum(seq_time_exploit))  # Eq 15
        
        # Calculate function value Eq 17 and 18
        if score_explore < score_exploit:
            mega_explor = max((mega_explor - 0.01), 0.01)
            mega_exploit = 0.99
        elif score_explore > score_exploit:
            mega_explor = 0.99
            mega_exploit = max((mega_exploit - 0.01), 0.01)
            
        lmn_explore = 1 - mega_explor  # Eq 24
        lmn_exploit = 1 - mega_exploit  # Eq 22
        
        score_explore = (mega_explor * f1_explor) + (mega_explor * f2_explor) + (lmn_explore * (min(pf_f3) * f3_explore))  # Eq 20
        score_exploit = (mega_exploit * f1_exploit) + (mega_exploit * f2_exploit) + (lmn_exploit * (min(pf_f3) * f3_exploit))  # Eq 19
        
        convergence.append(best.Cost)
        print(f'Iteration: {iter} Best Cost = {best.Cost}')
        
    return best.X, best.Cost, convergence

# Example test functions
def sphere(x: np.ndarray) -> float:
    """Sphere test function"""
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock test function"""
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0))

if __name__ == "__main__":
    # Example usage
    dim = 30
    n_sol = 50
    max_iter = 500
    lb = -100
    ub = 100
    
    best_x, best_cost, convergence = puma(n_sol, max_iter, lb, ub, dim, sphere)
    print(f"\nBest solution found: {best_x}")
    print(f"Best cost: {best_cost}") 