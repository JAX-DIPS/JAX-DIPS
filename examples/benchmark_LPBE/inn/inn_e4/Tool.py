# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from .min_norm_solvers_change import MinNormSolver


def grad(y, x):
    (dydx,) = torch.autograd.grad(
        outputs=y,
        inputs=x,
        retain_graph=True,
        grad_outputs=torch.ones(y.size()).to(y.device),
        create_graph=True,
        allow_unused=True,
    )
    return dydx


def data_transform(data):
    x = data[0].view(-1, 1)
    y = data[1].view(-1, 1)
    z = data[2].view(-1, 1)
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    input = torch.cat((x, y, z), 1)

    return x, y, z, input


def gradient(data, x, y, z):
    dx = grad(data, x)
    dy = grad(data, y)
    dz = grad(data, z)

    return dx, dy, dz


def MGDA_getparam(task_key, task, net_in, net_out, optim, device):
    if task_key != "outb":
        optim.zero_grad()
        task.backward(retain_graph=True)

        for i, param in enumerate(net_in.parameters()):
            if param.grad is not None:
                if i == 0:
                    gradsin = (
                        param.grad.data.clone()
                        .detach_()
                        .view(
                            -1,
                        )
                    )
                else:
                    gradsin = torch.cat(
                        [
                            gradsin,
                            param.grad.data.clone()
                            .detach_()
                            .view(
                                -1,
                            ),
                        ],
                        0,
                    )
            elif param.grad is None:
                if i == 0:
                    gradsin = torch.zeros_like(
                        param.grad.data.clone()
                        .detach_()
                        .view(
                            -1,
                        )
                    )
                else:
                    gradsin = torch.cat(
                        [
                            gradsin,
                            torch.zeros(1)
                            .view(
                                -1,
                            )
                            .to(device),
                        ],
                        0,
                    )

        gradsin = gradsin.view(1, -1)

        for i, param in enumerate(net_out.parameters()):
            if param.grad is not None:
                if i == 0:
                    gradso = (
                        param.grad.data.clone()
                        .detach_()
                        .view(
                            -1,
                        )
                    )
                else:
                    gradso = torch.cat(
                        [
                            gradso,
                            param.grad.data.clone()
                            .detach_()
                            .view(
                                -1,
                            ),
                        ],
                        0,
                    )
            elif param.grad is None:
                if i == 0:
                    gradso = torch.zeros_like(
                        param.grad.data.clone()
                        .detach_()
                        .view(
                            -1,
                        )
                    )
                else:
                    gradso = torch.cat(
                        [
                            gradso,
                            torch.zeros(1)
                            .view(
                                -1,
                            )
                            .to(device),
                        ],
                        0,
                    )

        gradso = gradso.view(1, -1)

        return gradsin, gradso

    else:
        optim.zero_grad()
        task.backward(retain_graph=True)

        for i, param in enumerate(net_out.parameters()):
            if param.grad is not None:
                if i == 0:
                    gradso = (
                        param.grad.data.clone()
                        .detach_()
                        .view(
                            -1,
                        )
                    )
                else:
                    gradso = torch.cat(
                        [
                            gradso,
                            param.grad.data.clone()
                            .detach_()
                            .view(
                                -1,
                            ),
                        ],
                        0,
                    )
            elif param.grad is None:
                if i == 0:
                    gradso = torch.zeros_like(
                        param.grad.data.clone()
                        .detach_()
                        .view(
                            -1,
                        )
                    )
                else:
                    gradso = torch.cat(
                        [
                            gradso,
                            torch.zeros(1)
                            .view(
                                -1,
                            )
                            .to(device),
                        ],
                        0,
                    )

        gradso = gradso.view(1, -1)

        return gradso


def MGDA_train(epoch, task, task_loss, net_in, net_out, optim, device, s, NN, c=1):
    grads_in = {}
    grads_out = {}

    grads_in["loss_in"], grads_out["loss_out"] = MGDA_getparam(
        "loss_in_add_out", task["loss_in_add_out"], net_in, net_out, optim, device
    )
    grads_in["loss_in"], grads_out["loss_out"] = (
        grads_in["loss_in"] / task_loss["1"],
        grads_out["loss_out"] / task_loss["4"],
    )

    w1, w2 = MGDA_getparam("bd_add_bn", task["bd_add_bn"], net_in, net_out, optim, device)
    grads_out["loss_boundary"] = MGDA_getparam("outb", task["outb"], net_in, net_out, optim, device)
    grads_out["loss_boundary"] = grads_out["loss_boundary"] / task_loss["5"]

    tin_scal, min_norm = MinNormSolver.find_min_norm_element_FW([w1, grads_in["loss_in"]])
    tout_scal, min_norm = MinNormSolver.find_min_norm_element_FW(
        [w2, grads_out["loss_out"], grads_out["loss_boundary"]]
    )

    tol = 1e6
    scale = {}

    if float(tout_scal[0]) < float(tin_scal[0]):
        scale["loss_in"] = np.min(
            (c * float(tin_scal[1]) * float(tout_scal[0]) / float(tin_scal[0]) / task_loss["1"], tol)
        )
        scale["loss_gammad"] = np.min((c * float(tout_scal[0]) * s / NN / task_loss["2"], tol))
        scale["loss_gamman"] = np.min((c * float(tout_scal[0]) * (1 - s / NN) / task_loss["3"], tol))
        scale["loss_out"] = np.min((c * float(tout_scal[1]) / task_loss["4"], tol))
        scale["loss_boundary"] = np.min((c * float(tout_scal[2]) / task_loss["5"], tol))
    else:
        scale["loss_in"] = np.min((c * float(tin_scal[1]) / task_loss["1"], tol))
        scale["loss_gammad"] = np.min((c * float(tin_scal[0]) * (s / NN) / task_loss["2"], tol))
        scale["loss_gamman"] = np.min((c * float(tin_scal[0]) * (1 - s / NN) / task_loss["3"], tol))
        scale["loss_out"] = np.min(
            (c * float(tout_scal[1]) * float(tin_scal[0]) / float(tout_scal[0]) / task_loss["4"], tol)
        )
        scale["loss_boundary"] = np.min(
            (c * float(tout_scal[2]) * float(tin_scal[0]) / float(tout_scal[0]) / task_loss["5"], tol)
        )

    return scale
