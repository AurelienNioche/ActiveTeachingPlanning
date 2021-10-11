import numpy as np
import torch
from torch import distributions as dist


class LossMinibatch:
    @staticmethod
    def __call__(z_flow, theta_flow, n_sample, n_u, n_w,
                 u, w, x, r, y):
        # Get unique users for this (mini)batch
        uniq_u = np.unique(u)
        uniq_w = np.unique(w)

        # Z: Sample base distribution and apply transformation
        z0_Z = z_flow.sample_base_dist(n_sample)
        zk_Z, ln_q0_Z, sum_ld_Z = z_flow(z0_Z)

        # θ: Sample base distribution and apply transformation
        z0_θ = theta_flow.sample_base_dist(n_sample)
        zk_θ, ln_q0_θ, sum_ld_θ = theta_flow(z0_θ)

        # Get Z-values used for first parameter
        Zu1 = zk_Z[:, :n_u].T
        Zw1 = zk_Z[:, n_u:n_w + n_u].T

        # Get Z-values used for second first parameter
        Zu2 = zk_Z[:, n_w + n_u:n_w + n_u * 2].T
        Zw2 = zk_Z[:, n_w + n_u * 2:].T

        # Get θ-values for both parameters
        mu1, log_var_u1, log_var_w1 = zk_θ[:, :3].T
        mu2, log_var_u2, log_var_w2 = zk_θ[:, 3:].T

        # Compute Z-values for both parameters
        Z1 = Zu1[u] + Zw1[w]
        Z2 = Zu2[u] + Zw2[w]

        # Go to constrained space
        param1 = torch.exp(Z1)
        param2 = torch.sigmoid(Z2)

        # Compute log probability of recall
        log_p = -param1 * x * (1 - param2) ** r

        # Comp. likelihood of observations
        ll = dist.Bernoulli(probs=torch.exp(log_p)).log_prob(y).sum(axis=0)

        # Comp. likelihood Z-values given population parameterization for first parameter
        ll_Zu1 = dist.Normal(mu1, torch.exp(0.5 * log_var_u1)).log_prob(
            Zu1[uniq_u]).sum(axis=0)
        ll_Zw1 = dist.Normal(mu1, torch.exp(0.5 * log_var_w1)).log_prob(
            Zw1[uniq_w]).sum(axis=0)

        # Comp. likelihood Z-values given population parameterization for second parameter
        ll_Zu2 = dist.Normal(mu2, torch.exp(0.5 * log_var_u2)).log_prob(
            Zu2[uniq_u]).sum(axis=0)
        ll_Zw2 = dist.Normal(mu2, torch.exp(0.5 * log_var_w2)).log_prob(
            Zw2[uniq_w]).sum(axis=0)

        # Add all the loss terms and compute average (= expectation estimate)
        to_min = (ln_q0_Z + ln_q0_θ
                  - sum_ld_Z - sum_ld_θ
                  - ll - ll_Zu1 - ll_Zu2 - ll_Zw1 - ll_Zw2).mean()
        return to_min