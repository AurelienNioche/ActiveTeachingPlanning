from inference.train import train
from inference.flows import NormalizingFlow
from simulate.simulate import simulate
from plot.plot_hist_loss import plot_loss
from plot.plot_posterior import plot_posterior

from data.get_experimental_data import get_experimental_data


def main():

    data = get_experimental_data()

    z_flow, theta_flow, hist_loss = train(
        data,
        n_sample=40,
        epochs=5000)
    z_flow.save("z_flow_exp_data")
    theta_flow.save("theta_flow_exp_data")
    # z_flow = NormalizingFlow.load("z_flow_artificial")
    # theta_flow = NormalizingFlow.load("theta_flow_artificial")
    plot_posterior(theta_flow=theta_flow,
                   name="artificial")
    plot_loss(hist_loss=hist_loss, name="artificial")


if __name__ == "__main__":
    main()
