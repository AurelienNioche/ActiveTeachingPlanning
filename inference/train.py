import numpy as np
import torch
from torch import optim
from tqdm.autonotebook import tqdm

from . flows import NormalizingFlows
from . loss import LossTeaching


def train(
        data,
        flow_length=16,
        epochs=5000,
        optimizer_name="Adam",
        optimizer_kwargs=None,
        initial_lr=0.01,
        n_sample=40,
        seed=123):

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_u = len(np.unique(data['u']))
    n_w = len(np.unique(data['w']))

    z_flow = NormalizingFlows(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlows(6, flow_length=flow_length)

    loss_func = LossTeaching()

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = getattr(optim, optimizer_name)(
            list(z_flow.parameters()) + list(theta_flow.parameters()),
            lr=initial_lr, **optimizer_kwargs)

    hist_loss = []

    with tqdm(total=epochs) as pbar:

        for i in range(epochs):

            optimizer.zero_grad()
            loss = loss_func(z_flow=z_flow,
                             theta_flow=theta_flow,
                             n_sample=n_sample,
                             n_u=n_u,
                             n_w=n_w,
                             **data)
            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())

            if i > 0:
                pbar.set_postfix({'loss': hist_loss[-1]})
            pbar.update()

    return z_flow, theta_flow, hist_loss


