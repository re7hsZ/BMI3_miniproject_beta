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

        ### ADDED: 在 HMM 类中新增随机初始化函数
    def initialize_random_params(self, observations, seed=12345):
        """
        If means/covars/bernoulli_params are None, initialize them from observations.
        observations: list of (comp_vec, context_vec)
        """
        T = len(observations)
        if T == 0:
            raise ValueError("No observations for initialization.")

        comp_array = np.vstack([obs[0] for obs in observations])
        n_features = comp_array.shape[1]
        self.n_features = n_features if self.n_features is None else self.n_features

        # simple k-means-like random partition initialization
        rng = np.random.default_rng(seed)
        assignments = rng.integers(0, self.n_states, size=T)

        # allocate means/covars if missing
        if self.means is None:
            self.means = np.zeros((self.n_states, n_features))
        if self.covars is None:
            self.covars = np.zeros((self.n_states, n_features, n_features))

        for j in range(self.n_states):
            sel = np.where(assignments == j)[0]
            if len(sel) == 0:
                sel = np.array([rng.integers(0, T)])
            subset = comp_array[sel]
            self.means[j] = np.mean(subset, axis=0)
            cov = np.cov(subset, rowvar=False)
            # ensure cov is 2D matrix
            if cov.ndim == 0:
                cov = np.eye(n_features) * float(cov)
            if cov.shape == (n_features,):
                cov = np.diag(cov)
            self.covars[j] = cov + np.eye(n_features) * 1e-6

        # bernoulli init from contexts if available
        if observations[0][1] is not None:
            ctx_array = np.vstack([obs[1] for obs in observations])
            if self.bernoulli_params is None:
                self.bernoulli_params = np.zeros((self.n_states, ctx_array.shape[1]))
            for j in range(self.n_states):
                sel = np.where(assignments == j)[0]
                if len(sel) == 0:
                    sel = np.array([0])
                self.bernoulli_params[j] = np.clip(np.mean(ctx_array[sel], axis=0), 1e-6, 1-1e-6)

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
    
        # ------------------- 新增：计算 log_xi（用于 EM） -------------------
    def _compute_log_xi(self, log_alpha, log_beta, observations, log_A):
        """
        Compute log_xi for t=0..T-2: log_xi[t,i,j] = log P(s_t=i, s_{t+1}=j | O)
        Inputs:
            log_alpha: (T, n_states)
            log_beta: (T, n_states)
            observations: list of (comp_vec, context_vec)
            log_A: log(self.A)
        Returns:
            log_xi: (T-1, n_states, n_states)
        """
        T = len(observations)
        n = self.n_states
        log_xi = np.full((T-1, n, n), -np.inf)
        # Precompute emissions for t+1
        for t in range(T-1):
            log_em_next = self._log_emission_prob(observations[t+1])  # shape (n_states,)
            # numerator: log_alpha[t,i] + log_A[i,j] + log_emission[t+1,j] + log_beta[t+1,j]
            # we compute for all i,j
            # use broadcasting
            num = (log_alpha[t][:, None] + log_A) + (log_em_next + log_beta[t+1])[None, :]
            # normalize by log P(O) at time t (same for all i,j)
            # log P(O) can be obtained by logsumexp(log_alpha[T-1]) but it's constant across t.
            # We'll normalize per t by subtracting logsumexp(num)
            den = self._logsumexp(num.ravel())
            log_xi[t] = num - den
        return log_xi

    # ------------------- added：Baum-Welch（EM）training -------------------
    def baum_welch_train(self, observations, max_iters=100, tol=1e-4, verbose=False):
        """
        Baum-Welch EM to refine A, pi, Gaussian (means,covars), and Bernoulli params.
        observations: list of (comp_vec, context_vec)
        """
        T = len(observations)
                # ----------------- MODIFIED: ensure parameters exist before EM -----------------
        # If means/covars/bernoulli_params not initialized, initialize from data
        if self.means is None or self.covars is None:
            if verbose:
                print("[BW] Info: means/covars not initialized - performing random initialization.")
            self.initialize_random_params(observations)
        if self.bernoulli_params is None and observations[0][1] is not None:
            # initialize bernoulli p with small values if missing
            if verbose:
                print("[BW] Info: bernoulli_params not initialized - estimating from data.")
            ctx_tmp = np.vstack([obs[1] for obs in observations])
            # overall mean per column
            self.bernoulli_params = np.clip(np.mean(ctx_tmp, axis=0)[None, :].repeat(self.n_states, axis=0), 1e-6, 1-1e-6)

        if T == 0:
            raise ValueError("No observations provided to baum_welch_train.")

        # Pre-allocate arrays for comp and context for easier vectorized updates
        comp_array = np.vstack([obs[0] for obs in observations])  # (T, n_features)
        ctx_array = None
        if observations[0][1] is not None:
            ctx_array = np.vstack([obs[1] for obs in observations])  # (T, n_ctx)

        prev_ll = -np.inf
        for iteration in range(1, max_iters + 1):
            # E-step: forward/backward
            log_alpha = self.forward_log(observations)
            log_beta = self.backward_log(observations)
            log_A = np.log(self.A + 1e-12)

            # log P(O)
            ll = self._logsumexp(log_alpha[-1])
            if verbose:
                print(f"[BW] Iter {iteration}  log-likelihood = {ll:.6f}")

            # compute gamma: P(state | O)
            log_gamma_unnorm = log_alpha + log_beta  # (T, n_states)
            # normalize per time step
            log_gamma = log_gamma_unnorm - np.expand_dims(np.apply_along_axis(self._logsumexp, 1, log_gamma_unnorm), axis=1)
            gamma = np.exp(log_gamma)  # (T, n_states)

            # compute xi (in log-space) for t=0..T-2
            log_xi = self._compute_log_xi(log_alpha, log_beta, observations, log_A)
            xi = np.exp(log_xi)  # shape (T-1, n, n)

            # M-step: reestimate parameters

            # 1) pi <- gamma[0]
            self.pi = np.clip(gamma[0], 1e-12, 1.0)
            self.pi = self.pi / np.sum(self.pi)

            # 2) A_ij <- sum_t xi_t(i,j) / sum_t gamma_t(i)  for t=0..T-2
            sum_xi = np.sum(xi, axis=0)  # shape (n, n)
            sum_gamma_except_last = np.sum(gamma[:-1], axis=0)  # shape (n,)
            # avoid division by zero
            for i in range(self.n_states):
                denom = sum_gamma_except_last[i] if sum_gamma_except_last[i] > 0 else 1e-12
                self.A[i, :] = sum_xi[i, :] / denom
            # Ensure rows sum to 1 and clip
            # ensure non-negative then normalize rows to sum to 1
            self.A = np.clip(self.A, 1e-12, None)
            row_sums = np.sum(self.A, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-12
            self.A = self.A / row_sums


            # 3) Gaussian params: means and covariances
            n_features = self.n_features
            for j in range(self.n_states):
                gamma_j = gamma[:, j]  # shape (T,)
                denom = np.sum(gamma_j)
                if denom < 1e-12:
                    # if no responsibility, skip update (keep previous params)
                    continue
                # weighted mean
                weighted_sum = np.sum(comp_array * gamma_j[:, None], axis=0)
                new_mean = weighted_sum / denom
                self.means[j] = new_mean

                # weighted covariance
                diffs = comp_array - new_mean[None, :]
                # compute weighted covariance
                cov = (diffs * gamma_j[:, None]).T @ diffs
                cov = cov / denom
                # regularize
                cov += np.eye(n_features) * 1e-6
                self.covars[j] = cov

            # 4) Bernoulli params
            if self.bernoulli_params is not None and ctx_array is not None:
                n_ctx = ctx_array.shape[1]
                for j in range(self.n_states):
                    gamma_j = gamma[:, j]  # (T,)
                    denom = np.sum(gamma_j)
                    if denom < 1e-12:
                        continue
                    # weighted average per context dimension
                    weighted = np.sum(ctx_array * gamma_j[:, None], axis=0)
                    p = weighted / denom
                    # clip away from 0/1 for stability
                    p = np.clip(p, 1e-6, 1 - 1e-6)
                    self.bernoulli_params[j] = p

            # Convergence check
            if np.isfinite(ll) and (abs(ll - prev_ll) < tol):
                if verbose:
                    print(f"[BW] Converged at iter {iteration}, ΔLL={abs(ll-prev_ll):.6e}")
                break
            prev_ll = ll

        return ll  # final log likelihood

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
