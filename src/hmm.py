import numpy as np

class HMM:
    def __init__(self, n_states=3, n_features=None, n_context_features=None):
        """
        Initialize HMM with 3 states: 0=Host, 1=Ameliorated, 2=Foreign.
        """
        self.n_states = n_states
        self.n_features = n_features
        self.n_context_features = n_context_features
        
        # Transition matrix (A)
        # Host <-> Ameliorated <-> Foreign
        # Favor self-transitions
        self.A = np.array([
            [0.95, 0.04, 0.01], # From Host
            [0.10, 0.80, 0.10], # From Ameliorated
            [0.01, 0.04, 0.95]  # From Foreign
        ])
        
        # Initial state probabilities (pi)
        self.pi = np.array([0.98, 0.01, 0.01])
        
        # Emission parameters
        # Channel 1: Composition (Gaussian)
        self.means = None
        self.covars = None
        
        # Channel 2: Context (Bernoulli)
        # Params: p (probability of 1) for each feature in each state
        # Shape: (n_states, n_context_features)
        self.bernoulli_params = None
        
        # Weight for context channel (scalar)
        self.context_weight = 1.0

    def _log_gaussian_pdf(self, x, mean, cov):
        """
        Calculate Log Gaussian PDF for a single observation x.
        """
        k = len(x)
        cov_reg = cov + np.eye(k) * 1e-6
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            return -np.inf
        inv = np.linalg.inv(cov_reg)
        x_mu = x - mean
        exponent = -0.5 * (x_mu.T @ inv @ x_mu)
        log_norm = -0.5 * (k * np.log(2*np.pi) + logdet)
        return log_norm + exponent

    def _log_bernoulli_pmf(self, x, p):
        """
        Calculate Log Bernoulli PMF for binary vector x and params p.
        log P(x|p) = sum(x_i * log(p_i) + (1-x_i) * log(1-p_i))
        """
        # Clip p to avoid log(0)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))

    def _log_emission_prob(self, observation):
        """
        Calculate log P(x | state) for all states.
        observation is tuple: (comp_vec, context_vec)
        """
        comp_vec, context_vec = observation
        log_probs = np.zeros(self.n_states)
        
        for i in range(self.n_states):
            # Channel 1: Gaussian
            log_gauss = self._log_gaussian_pdf(comp_vec, self.means[i], self.covars[i])
            
            # Channel 2: Bernoulli (if context available)
            log_bern = 0.0
            if self.bernoulli_params is not None and context_vec is not None:
                log_bern = self._log_bernoulli_pmf(context_vec, self.bernoulli_params[i])
            
            # Combine: log(P_gauss * P_bern^w) = log_gauss + w * log_bern
            log_probs[i] = log_gauss + self.context_weight * log_bern
            
        return log_probs

    def train_supervised(self, host_comp, host_context, foreign_comp, foreign_context):
        """
        Initialize parameters using labeled data.
        Host -> State 0
        Foreign -> State 2
        Ameliorated -> State 1 (Intermediate)
        """
        self.n_features = host_comp.shape[1]
        if host_context is not None:
            self.n_context_features = host_context.shape[1]
        
        self.means = np.zeros((self.n_states, self.n_features))
        self.covars = np.zeros((self.n_states, self.n_features, self.n_features))
        if self.n_context_features:
            self.bernoulli_params = np.zeros((self.n_states, self.n_context_features))
        
        # State 0: Host
        self.means[0] = np.mean(host_comp, axis=0)
        self.covars[0] = np.cov(host_comp, rowvar=False)
        if self.n_context_features:
            self.bernoulli_params[0] = np.mean(host_context, axis=0)
        
        # State 2: Foreign
        self.means[2] = np.mean(foreign_comp, axis=0)
        self.covars[2] = np.cov(foreign_comp, rowvar=False)
        if self.n_context_features:
            self.bernoulli_params[2] = np.mean(foreign_context, axis=0)
            
        # State 1: Ameliorated (Intermediate)
        self.means[1] = (self.means[0] + self.means[2]) / 2
        self.covars[1] = (self.covars[0] + self.covars[2]) / 2 + np.eye(self.n_features) * 1e-3
        if self.n_context_features:
            # Ameliorated context might be closer to Host or Foreign?
            # Let's assume intermediate
            self.bernoulli_params[1] = (self.bernoulli_params[0] + self.bernoulli_params[2]) / 2
        
        # Regularization
        for i in range(self.n_states):
            self.covars[i] += np.eye(self.n_features) * 1e-4

    def _logsumexp(self, a):
        """
        Compute log(sum(exp(a))) safely.
        """
        max_a = np.max(a)
        if np.isinf(max_a):
            return max_a
        return max_a + np.log(np.sum(np.exp(a - max_a)))

    def forward_log(self, observations):
        """
        Forward algorithm in log space.
        Returns:
            log_alpha: (T, n_states)
        """
        T = len(observations)
        log_alpha = np.zeros((T, self.n_states))
        
        # Initialization
        log_emission_0 = self._log_emission_prob(observations[0])
        log_alpha[0] = np.log(self.pi + 1e-10) + log_emission_0
        
        # Recursion
        log_A = np.log(self.A + 1e-10)
        
        for t in range(1, T):
            log_emission_t = self._log_emission_prob(observations[t])
            for j in range(self.n_states):
                # log(sum_i (alpha[t-1][i] * A[i][j]))
                # = logsumexp(log_alpha[t-1] + log_A[:, j])
                log_alpha[t, j] = self._logsumexp(log_alpha[t-1] + log_A[:, j]) + log_emission_t[j]
                
        return log_alpha

    def backward_log(self, observations):
        """
        Backward algorithm in log space.
        Returns:
            log_beta: (T, n_states)
        """
        T = len(observations)
        log_beta = np.zeros((T, self.n_states))
        
        # Initialization (log(1) = 0)
        log_beta[T-1] = 0.0
        
        # Recursion
        log_A = np.log(self.A + 1e-10)
        
        for t in range(T-2, -1, -1):
            log_emission_next = self._log_emission_prob(observations[t+1])
            for i in range(self.n_states):
                # log(sum_j (A[i][j] * emission[t+1][j] * beta[t+1][j]))
                term = log_A[i, :] + log_emission_next + log_beta[t+1]
                log_beta[t, i] = self._logsumexp(term)
                
        return log_beta

    def posterior(self, observations):
        """
        Calculate posterior probabilities P(state | observations).
        """
        log_alpha = self.forward_log(observations)
        log_beta = self.backward_log(observations)
        
        # log_gamma = log_alpha + log_beta - log_P(O)
        # log_P(O) = logsumexp(log_alpha[T-1])
        
        log_gamma_unnorm = log_alpha + log_beta
        
        # Normalize per time step
        gamma = np.zeros_like(log_gamma_unnorm)
        for t in range(len(observations)):
            log_norm = self._logsumexp(log_gamma_unnorm[t])
            gamma[t] = np.exp(log_gamma_unnorm[t] - log_norm)
            
        return gamma

    def viterbi(self, observations):
        """
        Viterbi algorithm for decoding most likely state sequence.
        Returns:
            path: List of state indices
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        log_emission_0 = self._log_emission_prob(observations[0])
        epsilon = 1e-10
        
        delta[0] = np.log(self.pi + epsilon) + log_emission_0
        
        # Recursion
        log_A = np.log(self.A + epsilon)
        
        for t in range(1, T):
            log_emission_t = self._log_emission_prob(observations[t])
            
            for j in range(self.n_states):
                # max_{i} (delta[t-1][i] + log_A[i][j])
                prev_scores = delta[t-1] + log_A[:, j]
                best_prev = np.argmax(prev_scores)
                psi[t, j] = best_prev
                delta[t, j] = prev_scores[best_prev] + log_emission_t[j]
                
        # Termination
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        
        # Backtracking
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        return path
