import numpy as np
import pandas as pd

from inference.train import train
from plot.inference.plot_hist_loss import plot_loss
from plot.inference.plot_posterior import plot_posterior

from data_preprocessing.user_data_preprocessing import get_preprocessed_data


def run_inference():
    data = get_preprocessed_data()

    z_flow, theta_flow, hist_loss = train(
        data,
        n_sample=40,
        epochs=5000)
    z_flow.save("z_flow_exp_data")
    theta_flow.save("theta_flow_exp_data")
    # z_flow = NormalizingFlow.load("z_flow_exp_data")
    # theta_flow = NormalizingFlow.load("theta_flow_exp_data")
    plot_posterior(theta_flow=theta_flow,
                   name="artificial")
    plot_loss(hist_loss=hist_loss, name="artificial")
    return theta_flow


def save_population_parameters(theta_flow, batch_size=int(10e5)):

    z0_θ = theta_flow.sample_base_dist(batch_size)
    zk_θ, base_dist_logprob_θ, log_det_θ = theta_flow(z0_θ)

    mu1, log_var_u1, log_var_w1 = zk_θ.data[:, :3].T
    mu2, log_var_u2, log_var_w2 = zk_θ.data[:, 3:].T

    unconstrained = {
        "mu1": mu1.mean().item(),
        "sigma_u1": np.exp(0.5 * log_var_u1.mean().item()),
        "sigma_w1": np.exp(0.5 * log_var_w1.mean().item()),
        "mu2": mu2.mean().item(),
        "sigma_u2": np.exp(0.5 * log_var_u2.mean().item()),
        "sigma_w2": np.exp(0.5 * log_var_w2.mean().item())}

    df_param = pd.DataFrame([unconstrained, ], index=["unconstrained", ])
    df_param.to_csv("data/param_exp_data.csv")


def main():

    theta_flow = run_inference()
    save_population_parameters(theta_flow)


if __name__ == "__main__":
    main()
