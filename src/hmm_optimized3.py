import numpy as np


class HMM:
    def __init__(self, n_states=3, n_features=None, n_context_features=None):
        """3-state HMM: 0=Host, 1=Ameliorated, 2=Foreign."""
        self.n_states = n_states
        self.n_features = n_features
        self.n_context_features = n_context_features
        self.A = np.array([
            [0.95, 0.04, 0.01],
            [0.10, 0.80, 0.10],
            [0.01, 0.04, 0.95],
        ])
        self.pi = np.array([0.98, 0.01, 0.01])
        self.means = None
        self.covars = None
        self.bernoulli_params = None
        self.context_weight = 1.0

    def _log_gaussian_pdf(self, x, mean, cov):
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
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))

    def _log_emission_prob(self, observation):
        comp_vec, context_vec = observation
        log_probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            log_gauss = self._log_gaussian_pdf(comp_vec, self.means[i], self.covars[i])
            log_bern = 0.0
            if self.bernoulli_params is not None and context_vec is not None:
                log_bern = self._log_bernoulli_pmf(context_vec, self.bernoulli_params[i])
            log_probs[i] = log_gauss + self.context_weight * log_bern
        return log_probs

    def train_supervised(self, host_comp, host_context, foreign_comp, foreign_context):
        self.n_features = host_comp.shape[1]
        if host_context is not None:
            self.n_context_features = host_context.shape[1]
        self.means = np.zeros((self.n_states, self.n_features))
        self.covars = np.zeros((self.n_states, self.n_features, self.n_features))
        if self.n_context_features:
            self.bernoulli_params = np.zeros((self.n_states, self.n_context_features))

        self.means[0] = np.mean(host_comp, axis=0)
        self.covars[0] = np.cov(host_comp, rowvar=False)
        if self.n_context_features:
            self.bernoulli_params[0] = np.mean(host_context, axis=0)

        self.means[2] = np.mean(foreign_comp, axis=0)
        self.covars[2] = np.cov(foreign_comp, rowvar=False)
        if self.n_context_features:
            self.bernoulli_params[2] = np.mean(foreign_context, axis=0)

        self.means[1] = (self.means[0] + self.means[2]) / 2
        self.covars[1] = (self.covars[0] + self.covars[2]) / 2 + np.eye(self.n_features) * 1e-3
        if self.n_context_features:
            self.bernoulli_params[1] = (self.bernoulli_params[0] + self.bernoulli_params[2]) / 2

        for i in range(self.n_states):
            self.covars[i] += np.eye(self.n_features) * 1e-4

    def initialize_random_params(self, observations, seed=12345):
        T = len(observations)
        if T == 0:
            raise ValueError("No observations for initialization.")
        comp_array = np.vstack([obs[0] for obs in observations])
        n_features = comp_array.shape[1]
        self.n_features = n_features if self.n_features is None else self.n_features
        rng = np.random.default_rng(seed)
        assignments = rng.integers(0, self.n_states, size=T)
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
            if cov.ndim == 0:
                cov = np.eye(n_features) * float(cov)
            if cov.shape == (n_features,):
                cov = np.diag(cov)
            self.covars[j] = cov + np.eye(n_features) * 1e-6
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
        max_a = np.max(a)
        if np.isinf(max_a):
            return max_a
        return max_a + np.log(np.sum(np.exp(a - max_a)))

    def forward_log(self, observations):
        T = len(observations)
        log_alpha = np.zeros((T, self.n_states))
        log_emission_0 = self._log_emission_prob(observations[0])
        log_alpha[0] = np.log(self.pi + 1e-10) + log_emission_0
        log_A = np.log(self.A + 1e-10)
        for t in range(1, T):
            log_emission_t = self._log_emission_prob(observations[t])
            for j in range(self.n_states):
                log_alpha[t, j] = self._logsumexp(log_alpha[t-1] + log_A[:, j]) + log_emission_t[j]
        return log_alpha

    def backward_log(self, observations):
        T = len(observations)
        log_beta = np.zeros((T, self.n_states))
        log_A = np.log(self.A + 1e-10)
        for t in range(T-2, -1, -1):
            log_emission_next = self._log_emission_prob(observations[t+1])
            for i in range(self.n_states):
                term = log_A[i, :] + log_emission_next + log_beta[t+1]
                log_beta[t, i] = self._logsumexp(term)
        return log_beta

    def posterior(self, observations):
        log_alpha = self.forward_log(observations)
        log_beta = self.backward_log(observations)
        log_gamma_unnorm = log_alpha + log_beta
        gamma = np.zeros_like(log_gamma_unnorm)
        for t in range(len(observations)):
            log_norm = self._logsumexp(log_gamma_unnorm[t])
            gamma[t] = np.exp(log_gamma_unnorm[t] - log_norm)
        return gamma

    def _compute_log_xi(self, log_alpha, log_beta, observations, log_A):
        T = len(observations)
        n = self.n_states
        log_xi = np.full((T-1, n, n), -np.inf)
        for t in range(T-1):
            log_em_next = self._log_emission_prob(observations[t+1])
            num = (log_alpha[t][:, None] + log_A) + (log_em_next + log_beta[t+1])[None, :]
            den = self._logsumexp(num.ravel())
            log_xi[t] = num - den
        return log_xi

    def baum_welch_train(self, observations, max_iters=100, tol=1e-4, verbose=False):
        T = len(observations)
        if self.means is None or self.covars is None:
            if verbose:
                print("[BW] Info: means/covars not initialized - performing random initialization.")
            self.initialize_random_params(observations)
        if self.bernoulli_params is None and observations[0][1] is not None:
            if verbose:
                print("[BW] Info: bernoulli_params not initialized - estimating from data.")
            ctx_tmp = np.vstack([obs[1] for obs in observations])
            self.bernoulli_params = np.clip(np.mean(ctx_tmp, axis=0)[None, :].repeat(self.n_states, axis=0), 1e-6, 1-1e-6)
        if T == 0:
            raise ValueError("No observations provided to baum_welch_train.")
        comp_array = np.vstack([obs[0] for obs in observations])
        ctx_array = None
        if observations[0][1] is not None:
            ctx_array = np.vstack([obs[1] for obs in observations])
        prev_ll = -np.inf
        for iteration in range(1, max_iters + 1):
            log_alpha = self.forward_log(observations)
            log_beta = self.backward_log(observations)
            log_A = np.log(self.A + 1e-12)
            ll = self._logsumexp(log_alpha[-1])
            if verbose:
                print(f"[BW] Iter {iteration}  log-likelihood = {ll:.6f}")
            log_gamma_unnorm = log_alpha + log_beta
            log_gamma = log_gamma_unnorm - np.expand_dims(np.apply_along_axis(self._logsumexp, 1, log_gamma_unnorm), axis=1)
            gamma = np.exp(log_gamma)
            log_xi = self._compute_log_xi(log_alpha, log_beta, observations, log_A)
            xi = np.exp(log_xi)
            self.pi = np.clip(gamma[0], 1e-12, 1.0)
            self.pi = self.pi / np.sum(self.pi)
            sum_xi = np.sum(xi, axis=0)
            sum_gamma_except_last = np.sum(gamma[:-1], axis=0)
            for i in range(self.n_states):
                denom = sum_gamma_except_last[i] if sum_gamma_except_last[i] > 0 else 1e-12
                self.A[i, :] = sum_xi[i, :] / denom
            self.A = np.clip(self.A, 1e-12, None)
            row_sums = np.sum(self.A, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-12
            self.A = self.A / row_sums
            n_features = self.n_features
            for j in range(self.n_states):
                gamma_j = gamma[:, j]
                denom = np.sum(gamma_j)
                if denom < 1e-12:
                    continue
                weighted_sum = np.sum(comp_array * gamma_j[:, None], axis=0)
                new_mean = weighted_sum / denom
                self.means[j] = new_mean
                diffs = comp_array - new_mean[None, :]
                cov = (diffs * gamma_j[:, None]).T @ diffs
                cov = cov / denom
                cov += np.eye(n_features) * 1e-6
                self.covars[j] = cov
            if self.bernoulli_params is not None and ctx_array is not None:
                n_ctx = ctx_array.shape[1]
                for j in range(self.n_states):
                    gamma_j = gamma[:, j]
                    denom = np.sum(gamma_j)
                    if denom < 1e-12:
                        continue
                    weighted = np.sum(ctx_array * gamma_j[:, None], axis=0)
                    p = weighted / denom
                    p = np.clip(p, 1e-6, 1 - 1e-6)
                    self.bernoulli_params[j] = p
            if np.isfinite(ll) and (abs(ll - prev_ll) < tol):
                if verbose:
                    print(f"[BW] Converged at iter {iteration}, ΔLL={abs(ll-prev_ll):.6e}")
                break
            prev_ll = ll
        return ll

    def viterbi(self, observations):
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        log_emission_0 = self._log_emission_prob(observations[0])
        epsilon = 1e-10
        delta[0] = np.log(self.pi + epsilon) + log_emission_0
        log_A = np.log(self.A + epsilon)
        for t in range(1, T):
            log_emission_t = self._log_emission_prob(observations[t])
            for j in range(self.n_states):
                prev_scores = delta[t-1] + log_A[:, j]
                best_prev = np.argmax(prev_scores)
                psi[t, j] = best_prev
                delta[t, j] = prev_scores[best_prev] + log_emission_t[j]
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path
