import numpy as np
import torch
from torch import optim

from . flows import NormalizingFlow
from . loss_minibatch import LossMinibatch
from plot.loss_tracker import LossTracker


def train_minibatch(
        data,
        n_epochs=5000,
        initial_lr=0.01,
        flow_length=16,
        n_sample=40,
        batch_size=None,
        constant_lr=False,
        optimizer_name="Adam",
        scheduler_name=None,
        seed=123,
        online_plot=False, # Plot online losses
        online_plot_freq_update=None, # only matters if online_plot is true
        online_plot_max_n=None): # only matters if online_plot is true

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_u = len(np.unique(data['u']))
    n_w = len(np.unique(data['w']))
    n_obs = len(data['u'])

    if batch_size is None:
        batch_size = n_obs

    z_flow = NormalizingFlow(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlow(6, flow_length=flow_length)

    optimizer = getattr(optim, optimizer_name)(
        list(z_flow.parameters()) + list(theta_flow.parameters()),
        lr=initial_lr)

    if scheduler_name is not None:
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer)
        assert constant_lr is False
    else:
        scheduler = None

    loss_func = LossMinibatch()

    indices = np.arange(n_obs)

    with LossTracker(
            total=n_epochs,
            online_plot=online_plot,
            online_plot_freq_update=online_plot_freq_update,
            online_plot_max_n=online_plot_max_n) as loss_tracker:

        for epoch in range(n_epochs):

            np.random.shuffle(indices)

            for i in range(0, n_obs, batch_size):

                idx_minibatch = indices[i:i + batch_size]

                d = {k: v[idx_minibatch] for k, v in data.items()}

                optimizer.zero_grad()
                loss = loss_func(z_flow=z_flow,
                                 theta_flow=theta_flow,
                                 n_sample=n_sample,
                                 n_u=n_u, n_w=n_w,
                                 **d)
                loss.backward()
                optimizer.step()

                loss_tracker.append(loss.item())

                if constant_lr:
                    for g in optimizer.param_groups:
                        g['lr'] = initial_lr

                if scheduler is not None:
                    scheduler.step()

            loss_tracker.update()

    return z_flow, theta_flow, loss_tracker.hist_loss
