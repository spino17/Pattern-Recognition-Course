import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

"""
left to right Hidden Markov model for speech data

"""

class HMM:
    
    def __init__(self, N, M, x_train):
        self.N = N # no. of states
        self.M = M # no. of observation - no. of cluster used for K-means
        self.x_train = x_train # training dataset
        
        self.Pi = np.zeros(N) # initial probabilities
        self.Pi[0] = 1 # initial probability for starting state
        
        self.A = np.zeros((N, N)) # state transition matrix
        self.B = np.zeros((N, M)) # state observation probabilities
        total_size = 0
        for x_vec in x_train:
            x_vec = np.array(x_vec)
            epsilon_matrix = np.zeros((N, N))
            gamma_matrix = np.zeros((N, M))
            size = x_vec.size
            quotient = int(size / N)
            remainder = size % N
            
            # calculation for A matrix
            for i in range(0, N):
                for j in range(0, N):
                    if j < i:
                        epsilon_matrix[i][j] = 0
                    elif i == j:
                        if j < N - 1:
                            epsilon_matrix[i][j] = (quotient - 1) / quotient
                        else:
                            epsilon_matrix[i][j] = 1
                    else:
                        if j == i + 1:
                            epsilon_matrix[i][j] = 1 / quotient
                        else:
                            epsilon_matrix[i][j] = 0
                            
            self.A = self.A + epsilon_matrix
            total_size = total_size + 1
            
            # calculation for B matrix
            iter_num = 0
            state = 0
            for i in range(0, size):
                """if (iter_num < quotient):
                    gamma_matrix[state][x_vec[i]] = gamma_matrix[state][x_vec[i]] + 1
                    iter_num = iter_num + 1
                else:
                    iter_num = 0
                    i = i - 1
                    gamma_matrix[state] = gamma_matrix[state] / quotient
                    state = state + 1
                """
                if state < N - 1:
                    if iter_num < quotient:
                        gamma_matrix[state][x_vec[i]] = gamma_matrix[state][x_vec[i]] + 1
                        iter_num = iter_num + 1
                    else:
                        iter_num = 0
                        i = i - 1
                        gamma_matrix[state] = gamma_matrix[state] / quotient
                        state = state + 1
                else:
                    gamma_matrix[state][x_vec[i]] = gamma_matrix[state][x_vec[i]] + 1
                    
            gamma_matrix[N - 1] = gamma_matrix[N - 1] / (quotient + remainder)
            self.B = self.B + gamma_matrix
        
        self.A = (1 / total_size) * self.A
        self.B = (1 / total_size) * self.B
        self.total_size = total_size
        
        
    
    # returs optimal state sequence for the given observation sequence
    def state_seq(self, x_vec):
        x_vec = np.array(x_vec)
        Pi = self.Pi
        A = self.A
        b = self.B
        N = self.N
        
        size = x_vec.size
        current_col = np.zeros(N)
        
        # final state sequnece which has the maximum probability according to the HMM model
        psi = np.zeros(N, size)
        delta_col = np.zeros(N)
        state_seq = np.zeros(size)
        
        for i in range(0, N):
            current_col[i] = math.log(Pi[i]) + b[i][x_vec[0]]
            psi[i][0] = 0
        
        for t in range(1, size):
            
            delta_col = current_col
            for j in range(0, N):
                max_delta = delta_col[0] + math.log(A[0][j])
                max_psi = 0
                for i in range(0, N):
                    if max_delta <= (delta_col[i] + math.log(A[i][j])):
                        max_delta = delta_col[i] + math.log(A[i][j])
                        max_psi = i
                
                psi[j][t] = max_psi
                current_col[j] = max_delta + math.log(b[j][x_vec[t]])
        
        max_delta = current_col[0]
        state_seq[size - 1] = 0
        for i in range(0, N):
            if max_delta <= current_col[i]:
                max_delta = current_col[i]
                state_seq[size - 1] = i
        
        for time in range(1, size):
            t = size - time - 1
            state_seq[t] = psi[t + 1][state_seq[t + 1]]
        
        return state_seq
        
        
    # estimate the parameters of HMM using training dataset
    def fit(self, precision):
        
        N = self.N
        M = self.M
        x_train = self.x_train
        total_size = self.total_size
        
        iter_num = 0
        cost_func_i = 0
        cost_func_f = 0
        
        iter_axis = []
        cost_func_axis = []
        
        A_new = np.zeros((N, N))
        b_new = np.zeros((N, M))
        Pi_new = np.zeros(N)
        
        
        while iter_num < 3 or abs(cost_func_f - cost_func_i) > precision:
            
            Pi = self.Pi
            A = self.A
            b = self.B
            if iter_num == 1:
                max_likelihood = cost_func_f
                for i in range(0, N):
                    self.Pi[i] = Pi_new[i]
                for j in range(0, N):
                    self.A[i][j] = A_new[i][j]
                for j in range(0, M):
                    self.B[i][j] = b_new[i][j]
                Pi = Pi_new
                A = A_new
                b = b_new
                
            elif iter_num > 1:
                if cost_func_f > max_likelihood:
                    max_likelihood = cost_func_f
                    for i in range(0, N):
                        self.Pi[i] = Pi_new[i]
                    for j in range(0, N):
                        self.A[i][j] = A_new[i][j]
                    for j in range(0, M):
                        self.B[i][j] = b_new[i][j]
                Pi = Pi_new
                A = A_new
                b = b_new
            
            
            cost_func_i = cost_func_f
            cost_func_f = 0
            
            A_new = np.zeros((N, N))
            b_new = np.zeros((N, M))
            Pi_new = np.zeros(N)
            
            L = 0
            for x_vec in x_train:
                # runs for a specific data point
                
                #total_epsilon = 0
                #total_gamma = 0
                x_vec = np.array(x_vec)
                T = x_vec.size - 1
                beta_matrix = np.zeros((N, x_vec.size))
                alpha_col = np.zeros(N)
                current_col = np.zeros(N)
                epsilon_matrix = np.zeros((N, N))
                gamma_col = np.zeros(N)
                gamma_matrix = np.zeros((N, M))
                
                # code to calculate beta matrix
                for j in range(0, N):
                    beta_matrix[j][T] = 1
                    for i in range(0, N):
                        current_col[i] = current_col[i] + beta_matrix[j][T] * b[j][x_vec[T]] * A[i][j]
                
                for time in range(1, T):
                    t = T - time
                    beta_matrix[:, t] = current_col
                    current_col = np.zeros(N)
                    
                    for j in range(0, N):
                        for i in range(0 , N):
                            current_col[i] = current_col[i] + beta_matrix[j][t] * b[j][x_vec[t]] * A[i][j]
                
                beta_matrix[:, 0] = current_col
                #print(beta_matrix)
                
                # calculate P_o
                """p_o_beta = 0
                for j in range(0, N):
                    p_o_beta = p_o_beta + Pi[j] * b[j][x_vec[0]] * beta_matrix[j][0]
                """                
                current_col = np.zeros(N)
                
                # M-step started
                
                # initialization
                for i in range(0, N):
                    alpha_col[i] = Pi[i] * b[i][x_vec[0]]
                    for j in range(0, N):
                        current_col[j] = current_col[j] + alpha_col[i] * A[i][j]
                        
                for j in range(0, N):
                    current_col[j] = current_col[j] * b[j][x_vec[1]]
                
                
                total_gamma_prob = 0
                total_epsilon_prob = 0
                
                for i in range(0, N):
                    total_gamma_prob = total_gamma_prob + alpha_col[i] * beta_matrix[i][0]
                    for j in range(0, N):
                        total_epsilon_prob = total_epsilon_prob + alpha_col[i] * A[i][j] * b[j][x_vec[1]] * beta_matrix[j][1]
                        
                for i in range(0, N):
                    gamma_col[i] = gamma_col[i] + (alpha_col[i] * beta_matrix[i][0] / total_gamma_prob)
                    gamma_matrix[i][x_vec[0]] = gamma_matrix[i][x_vec[0]] + (alpha_col[i] * beta_matrix[i][0] / total_gamma_prob)
                    Pi_new[i] = Pi_new[i] + (alpha_col[i] * beta_matrix[i][0] / total_gamma_prob)
                    for j in range(0, N):
                        epsilon_matrix[i][j] = epsilon_matrix[i][j] + (alpha_col[i] * A[i][j] * b[j][x_vec[1]] * beta_matrix[j][1] / total_epsilon_prob)
                
                # induction
                for time in range(1, T):
                    t = time
                    alpha_col = current_col
                    current_col = np.zeros(N)
                    
                    for i in range(0, N):
                        for j in range(0, N):
                            current_col[j] = current_col[j] + alpha_col[i]  * A[i][j]
                    for j in range(0, N):
                        current_col[j] = current_col[j] * b[j][x_vec[t + 1]]
                        
                    total_epsilon_prob = 0
                    total_gamma_prob = 0
                    
                    for i in range(0, N):
                        total_gamma_prob = total_gamma_prob + alpha_col[i] * beta_matrix[i][t]
                        for j in range(0, N):
                            total_epsilon_prob = total_epsilon_prob + alpha_col[i] * A[i][j] * b[j][x_vec[t + 1]] * beta_matrix[j][t + 1]
                    
                    for i in range(0, N):
                        gamma_col[i] = gamma_col[i] + (alpha_col[i] * beta_matrix[i][t] / total_gamma_prob)
                        gamma_matrix[i][x_vec[t]] = gamma_matrix[i][x_vec[t]] + (alpha_col[i] * beta_matrix[i][t] / total_gamma_prob)
                        for j in range(0, N):
                            epsilon_matrix[i][j] = epsilon_matrix[i][j] + (alpha_col[i] * A[i][j] * b[j][x_vec[t + 1]] * beta_matrix[j][t + 1] / total_epsilon_prob)
                
                for i in range(0, N):
                    for j in range(0, N):
                        #A_new[i][j] = A_new[i][j] + (epsilon_matrix[i][j] / gamma_col[i])
                        if gamma_col[i] != 0:
                            A_new[i][j] = A_new[i][j] + (epsilon_matrix[i][j] / gamma_col[i])
                
                
                # calculation of P_o
                """p_o_alpha = 0
                for i in range(0, N):
                    p_o_alpha = p_o_alpha + current_col[i]"""
                
                # termination
                total_gamma_prob = 0
                P_o = 0
                for i in range(0, N):
                    P_o = P_o + current_col[i]
                    total_gamma_prob = total_gamma_prob + current_col[i] * beta_matrix[i][T]
                
                for i in range(0, N):
                    gamma_col[i] = gamma_col[i] + (current_col[i] * beta_matrix[i][T] / total_gamma_prob)
                    gamma_matrix[i][x_vec[T]] = gamma_matrix[i][x_vec[T]] + (current_col[i] * beta_matrix[i][T] / total_gamma_prob)
                
                for i in range(0, N):
                    for j in range(0, M):
                        #b_new[i][j] = b_new[i][j] + (gamma_matrix[i][j] / gamma_col[i])
                        if gamma_col[i] != 0:
                            b_new[i][j] = b_new[i][j] + (gamma_matrix[i][j] / gamma_col[i])
                
                L = L + 1
                cost_func_f = cost_func_f + math.log(P_o)
            
            
            Pi_new = (1 / total_size) * Pi_new
            A_new = (1 / total_size) * A_new
            b_new = (1 / total_size) * b_new
                                                                  
            # update the parameters with the new estimated parameters
            """self.A = A_new
            self.B = b_new
            self.Pi = Pi_new"""
            """
            for i in range(0, N):
                self.Pi[i] = Pi_new[i]
                for j in range(0, N):
                    self.A[i][j] = A_new[i][j]
                for j in range(0, M):
                    self.B[i][j] = b_new[i][j]"""
            
            
            iter_num = iter_num + 1
            
            print("iter no.: %d diff: %f cost function %f" % (iter_num, abs(cost_func_f - cost_func_i), cost_func_f))
            
            iter_axis.append(iter_num)
            cost_func_axis.append(cost_func_f)
        
        # plot the log likelihood vs iteration plot 
        plt.scatter(iter_axis, cost_func_axis)
        
    
    # gives the log likelihood for a sequence passed as arguments
    def score_alpha(self, x_vec):
        x_vec = np.array(x_vec)
        Pi = self.Pi
        A = self.A
        b = self.B
        size = x_vec.size
        N = self.N
        
        alpha_col = np.zeros(N)
        current_col = np.zeros(N)
        
        # initialization
        for i in range(0, N):
            alpha_col[i] = Pi[i] * b[i][x_vec[0]]
            for j in range(0, N):
                current_col[j] = current_col[j] + alpha_col[i] * A[i][j]
                
        for j in range(0, N):
            current_col[j] = current_col[j] * b[j][x_vec[1]]
            
        # induction
        for t in range(1, size - 1):
            alpha_col = current_col
            current_col = np.zeros(N)
            for i in range(0, N):
                for j in range(0, N):
                    current_col[j] = current_col[j] + alpha_col[i] * A[i][j]
            
            for j in range(0, N):
                current_col[j] = current_col[j] * b[j][x_vec[t + 1]]
                
        # termination
        score = 0            
        for i in range(0, N):
            score = score + current_col[i]
        
        return score
        
        
        
            
        
        
        