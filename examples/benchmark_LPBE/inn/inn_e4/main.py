import numpy as np
import argparse
import torch
import time, os
import itertools
import random
import torch.optim as optim
from Tool import grad, MGDA_train, data_transform, gradient
from Net_type import DeepRitzNet

from GenerateData import Data

################ LPBE ###############
alpha = 7.0465 * 1e3
kbT = 0.592783
pi = torch.tensor(np.pi)
cs = 80
cp = 2
Is = 1e-5
k2 = 8.4869 * Is / cs
k = float(np.sqrt(k2))


def g(x):
    x_norm = torch.norm(x, dim=1)
    return (alpha / (4 * pi * cp * x_norm)).view(-1, 1)


def true_solution_net(x, r0, label="inner", device="cpu"):
    x_norm = torch.norm(x, dim=1)
    r0 = torch.tensor(r0).to(device)
    if label == "inner":
        shape = torch.ones_like(x_norm).to(device)
        return (alpha / (4 * pi * r0) * (1 / (cs * (1 + k * r0)) - 1 / cp) * shape).view(-1, 1)
    elif label == "out":
        return (alpha / (4 * pi * cs) * torch.exp(k * r0) / (1 + k * r0) * torch.exp(-k * x_norm) / x_norm).view(-1, 1)


def inner_or_not(input, args):
    """
    Omega_1 or Omega_2
    """
    X_norm = np.linalg.norm(input - args.L, axis=1, ord=2)
    index_inner = np.where(X_norm < args.r0)[0]
    index_out = np.where(X_norm >= args.r0)[0]

    return index_inner, index_out


def test_data_net(device, args):
    r0 = args.r0
    h = 0.125
    X = np.arange(-2.5, 2.5 + h, h)
    Y = np.arange(-2.5, 2.5 + h, h)
    Z = np.arange(-2.5, 2.5 + h, h)
    X, Y, Z = np.meshgrid(X, Y, Z)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = Z.reshape(-1, 1)
    input = np.hstack((X, Y, Z))
    index_inner, index_out = inner_or_not(input, args)
    input = torch.tensor(input).float()
    inner = input[index_inner, :]
    test_inner = inner.float().to(device).clone().detach()
    label_inner = true_solution_net(test_inner, r0, "inner", device).clone().detach()

    out = input[index_out, :]
    test_out = out.float().to(device).clone().detach()
    label_out = true_solution_net(test_out, r0, "out", device).clone().detach()

    return test_out, label_out, test_inner, label_inner


def main(args):
    if torch.cuda.is_available and args.cuda:
        device = "cuda"
        print("cuda is avaliable")
    else:
        device = "cpu"

    ### test data
    test_out, label_out, test_inner, label_inner = test_data_net(device, args)

    ### train data
    data = Data(r0=args.r0, L=args.L, box=args.box, device=device)
    out = data.SampleFromOut(args.train_out).T
    x_out, y_out, z_out, input_out = data_transform(out)

    inner = data.SampleFromInner(args.train_inner).T
    x_in, y_in, z_in, input_in = data_transform(inner)

    gamma = data.SampleFromGamma(args.train_gamma).T
    x_gamma, y_gamma, z_gamma, input_in_b = data_transform(gamma)

    input_boundary = data.SampleFromBoundary(args.train_boundary)
    input_boundary_label = true_solution_net(input_boundary, args.r0, "out", device)

    g_D = g(input_in_b)
    f_direction = input_in_b.clone().detach_().to(device) / args.r0
    z = torch.ones(g_D.size()).to(device)
    gd = torch.autograd.grad(g_D, input_in_b, grad_outputs=z, create_graph=True)[0]
    g_N = (gd * f_direction).sum(dim=1).view(-1, 1) * cp

    net_inner = DeepRitzNet(m=args.inner_unit).to(device)
    net_out = DeepRitzNet(m=args.out_unit).to(device)
    optimizer = optim.Adam(itertools.chain(net_inner.parameters(), net_out.parameters()), lr=args.lr)
    result = []
    t0 = time.time()
    task = {}
    task_loss = {}
    loss_history = []
    if not os.path.isdir("./outputs/" + args.filename + "/model"):
        os.makedirs("./outputs/" + args.filename + "/model")
    print("Training...")

    Traing_Mse_min = 1e10
    Traing_Mse_min_epoch = 0
    for epoch in range(args.nepochs):
        optimizer.zero_grad()
        U1 = net_inner(input_in)
        U_1x, U_1y, U_1z = gradient(U1, x_in, y_in, z_in)
        U_1xx = grad(U_1x, x_in)
        U_1yy = grad(U_1y, y_in)
        U_1zz = grad(U_1z, z_in)
        loss_in = torch.mean((U_1xx + U_1yy + U_1zz) ** 2)

        U1_b = net_inner(input_in_b)
        U2_b_in = net_out(input_in_b)
        loss_gammad = torch.mean((U1_b + g_D - U2_b_in) ** 2)

        dU1_N = torch.autograd.grad(U1_b, input_in_b, grad_outputs=z, create_graph=True)[0]
        U1_N = (dU1_N * f_direction).sum(dim=1).view(-1, 1) * cp
        dU2_N = torch.autograd.grad(U2_b_in, input_in_b, grad_outputs=z, create_graph=True)[0]
        U2_N = (dU2_N * f_direction).sum(dim=1).view(-1, 1) * cs
        loss_gamman = torch.mean((g_N + U1_N - U2_N) ** 2)

        U2 = net_out(input_out)
        U_2x, U_2y, U_2z = gradient(U2, x_out, y_out, z_out)
        U_2xx = grad(U_2x, x_out)
        U_2yy = grad(U_2y, y_out)
        U_2zz = grad(U_2z, z_out)
        loss_out = torch.mean((-(U_2xx + U_2yy + U_2zz) + k2 * U2) ** 2)
        loss_boundary = torch.mean((net_out(input_boundary) - input_boundary_label) ** 2)

        ###########  INN
        NN = 3
        s = random.sample(range(0, NN), 1)[0]
        task["bd_add_bn"] = s / NN * loss_gammad / loss_gammad.data + (1 - s / NN) * loss_gamman / loss_gamman.data
        task["loss_in_add_out"] = loss_in + loss_out
        task["outb"] = loss_boundary

        task_loss["1"] = loss_in.item()
        task_loss["2"] = loss_gammad.item()
        task_loss["3"] = loss_gamman.item()
        task_loss["4"] = loss_out.item()
        task_loss["5"] = loss_boundary.item()

        scale = MGDA_train(epoch, task, task_loss, net_inner, net_out, optimizer, device, s, NN, c=1)
        loss = (
            loss_in * scale["loss_in"]
            + scale["loss_gammad"] * loss_gammad
            + loss_gamman * scale["loss_gamman"]
            + scale["loss_out"] * loss_out
            + scale["loss_boundary"] * loss_boundary
        )
        loss.backward(retain_graph=True)
        optimizer.step()

        # resample training data
        out = data.SampleFromOut(args.train_out).T
        x_out, y_out, z_out, input_out = data_transform(out)
        inner = data.SampleFromInner(args.train_inner).T
        x_in, y_in, z_in, input_in = data_transform(inner)
        gamma = data.SampleFromGamma(args.train_gamma).T
        x_gamma, y_gamma, z_gamma, input_in_b = data_transform(gamma)
        input_boundary = data.SampleFromBoundary(args.train_boundary)
        input_boundary_label = true_solution_net(input_boundary, args.r0, "out", device)
        g_D = g(input_in_b)
        f_direction = input_in_b.clone().detach_().to(device) / args.r0
        z = torch.ones(g_D.size()).to(device)
        gd = torch.autograd.grad(g_D, input_in_b, grad_outputs=z, create_graph=True)[0]
        g_N = (gd * f_direction).sum(dim=1).view(-1, 1) * cp

        if (epoch + 1) % args.print_num == 0:
            if (epoch + 1) % args.change_epoch == 0 and optimizer.param_groups[0]["lr"] > 1e-6:
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2

            with torch.no_grad():
                Mse_train = (loss_in + loss_out + loss_boundary + loss_gammad + loss_gamman).item()
                print("Epoch,  Training MSE: ", epoch + 1, Mse_train)
                print(
                    "Gamma_d,Gamma_n,Omega1,Omega2,partial_Omega: ",
                    loss_gammad.item(),
                    loss_gamman.item(),
                    loss_in.item(),
                    loss_out.item(),
                    loss_boundary.item(),
                )
                print("*****************************************************")

                if epoch > args.nepochs * 0.95:
                    torch.save(net_inner, "outputs/" + args.filename + "/model/{}net_inner.pkl".format(epoch))
                    torch.save(net_out, "outputs/" + args.filename + "/model/{}net_out.pkl".format(epoch))
                    if Traing_Mse_min > Mse_train:
                        Traing_Mse_min = Mse_train
                        Traing_Mse_min_epoch = epoch

                if args.save:
                    loss_history.append(
                        [
                            epoch,
                            loss_in.item(),
                            loss_gammad.item(),
                            loss_out.item(),
                            loss_boundary.item(),
                            loss_gamman.item(),
                        ]
                    )
                    loss_record = np.array(loss_history)
                    np.savetxt("outputs/" + args.filename + "/loss_record.txt", loss_record)

    print("_______________________________________________________________________________")
    print("_______________________________________________________________________________")
    print("Training min MSE:", Traing_Mse_min)
    print("The epoch of the training min MSE:", Traing_Mse_min_epoch + 1)
    pkl_in = "outputs/" + args.filename + "/model/{}net_inner.pkl".format(Traing_Mse_min_epoch)
    pkl_out = "outputs/" + args.filename + "/model/{}net_out.pkl".format(Traing_Mse_min_epoch)
    net_inner = torch.load(pkl_in)
    net_out = torch.load(pkl_out)
    # rela L_2
    L2_inner_loss = torch.sqrt(((net_inner(test_inner) - label_inner) ** 2).sum() / ((label_inner) ** 2).sum())
    L2_out_loss = torch.sqrt(((net_out(test_out) - label_out) ** 2).sum() / ((label_out) ** 2).sum())
    # L_infty
    L_inf_inner_loss = torch.max(torch.abs(net_inner(test_inner) - label_inner))
    L_inf_out_loss = torch.max(torch.abs(net_out(test_out) - label_out))
    print("L_infty:", max(L_inf_inner_loss.item(), L_inf_out_loss.item()))
    print("Rel. L_2:", (L2_inner_loss.item() * test_inner.size()[0] + L2_out_loss.item() * test_out.size()[0]) / 68921)

    print("totle use time:", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="result")
    parser.add_argument("--train_inner", type=int, default=100)
    parser.add_argument("--inner_unit", type=int, default=40)
    parser.add_argument("--out_unit", type=int, default=80)
    parser.add_argument("--train_gamma", type=int, default=200)
    parser.add_argument("--train_out", type=int, default=2000)
    parser.add_argument("--train_boundary", type=int, default=1000)
    parser.add_argument("--print_num", type=int, default=100)
    parser.add_argument("--nepochs", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", type=str, default=True)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--L", type=list, default=[0, 0, 0])
    parser.add_argument("--box", type=list, default=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5])
    parser.add_argument("--change_epoch", type=int, default=2000)
    parser.add_argument("--save", type=str, default=False)

    args = parser.parse_args()
    main(args)
