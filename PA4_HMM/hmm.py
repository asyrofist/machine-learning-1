from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | model)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        
        ###################################################
        obs_labels, obs_idx, obs_counts = np.unique(Osequence, return_index=True, return_counts=True)
        o = obs_counts / L
        self.o = obs_idx
        for t in range(L):
            for j in range(S):
                if t == 0:
                    alpha[j, t] = self.pi[j]
                else:
                    for s in range(S):
                        alpha[j, t] = alpha[j, t] + alpha[s][t - 1] * self.A[s, j]
                alpha[j, t] = alpha[j, t] * self.B[j, self.obs_dict[Osequence[t]]] 
        ###################################################
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, model)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:, L - 1] = 1
        for l in range(L - 2, -1, -1):
            for s in range(S):
                for k in range(S):
                    beta[s, l] = beta[s, l] + beta[k, l + 1] * self.A[s, k] * self.B[k, self.obs_dict[Osequence[l + 1]]]
        
        
        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | model)
        """
        prob = 0
        ###################################################
        # forward
        alpha = self.forward(Osequence)
        N = alpha.shape[1]
        # prob = np.sum(alpha[:, 0:N - 1])
        prob = np.sum(alpha[:, N - 1], axis=0)

        check_prob = 0
        # backward
        beta = self.backward(Osequence)
        S = beta.shape[0]
        for j in range(S):
            check_prob = check_prob + beta[j, 0] * self.pi[j] * self.B[j, self.obs_dict[Osequence[0]]]
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, model)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        pO = self.sequence_prob(Osequence)
        prob = (alpha * beta) / pO
        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        
        S = len(self.pi)
        N = len(Osequence)
        Q = np.zeros((S, N))
        # print('S:{} \n N: {}'.format(S,N))
        best_pred = np.zeros((S, N), dtype=int)

        for j in range(S):
            Q[j, 0] = self.pi[j] * self.B[j, self.obs_dict[Osequence[0]]]
        for t in range(1, N):
            for j in range(S):
                Q[j, t] = 0
                best_pred[j, t] = -1
                best_score = -np.inf
                for k in range(S):
                    # print(self.obs_dict[Osequence[t]])
                    # print(Q[k, t - 1)
                    r = self.A[k, j] * self.B[j, self.obs_dict[Osequence[t]]] + Q[k, t - 1]
                    if r > best_score:
                        best_score = r
                        best_pred[j, t] = int(k)
                        Q[j, t] = r

        final_best = -1
        final_score = -np.inf

        for j in range(S):
            if Q[j, N - 1] > final_score:
                final_score = Q[j, N - 1]
                final_best = int(j)

        current = final_best
        path = [current] + path

        for t in range(N - 2, -1, -1):
            path = [best_pred[current, t + 1]] + path
            current = best_pred[current, t + 1]
        # print('path: ',path)
        idx_state_dict = {v: k for k, v in self.state_dict.items()}
        path = [str(idx_state_dict[i]) for i in path]
        ###################################################
        return path
    
    def update_emissions(self, state_idx):
        new_obs = np.zeros((self.B.shape[0], 1))
        new_obs[state_idx] = 10 ** -6
        # print ('prior b shape',self.B.shape)
        self.B = np.append(self.B, new_obs, axis=1)
        # print(self.B.shape)
        # self.B[state_idx,obs_idx]=10**-6
        # Bs=self.B[state_idx]
        # bs
        return
