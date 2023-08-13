import numpy as np
import argparse
import torch
import time, os
import itertools
import random
import torch.optim as optim
from Tool import grad, MGDA_train, data_transform, gradient
from Net_type import DeepRitzNet

from GenerateData import Data, read_pqr


################## exact solution ##################
def u(x, label, b, device):
    """
    a=b=[1,100] inner out
    """
    x = x.t()
    if label == "inner":
        u = (torch.exp(x[0] * x[1] * x[2])).view(-1, 1)
    elif label == "out":
        u = (torch.sin(x[0] + x[1] + x[2])).view(-1, 1)
    elif label == "out_numan":
        u = b[1] * (torch.cos(x[0] + x[1] + x[2])).view(-1, 1).to(device) * (torch.tensor([1.0, 1.0, 1.0])).to(device)
    elif label == "inner_numan":
        u = b[0] * (
            torch.cat(((x[1] * x[2]).view(-1, 1), (x[0] * x[2]).view(-1, 1), (x[1] * x[0]).view(-1, 1)), 1)
            * (torch.exp(x[0] * x[1] * x[2])).view(-1, 1)
        ).to(device)
    else:
        raise ValueError("invalid label for u(x)")

    return u


def f_grad(x, label, a, b, device):
    xt = x.t()
    if label == "inner":
        f = (b[0] - a[0] * ((xt[0] * xt[1]) ** 2 + (xt[0] * xt[2]) ** 2 + (xt[2] * xt[1]) ** 2)).view(-1, 1) * u(
            x, label, a, device
        )
    elif label == "out":
        f = (3 * a[1] + b[1]) * u(x, label, a, device)
    else:
        raise ValueError("invalid label")

    return f


def phi_grad(x, f_direction, a, device):
    f = (u(x, "out_numan", a, device) - u(x, "inner_numan", a, device)) * f_direction
    p_grad = (f.sum(1)).view(-1, 1)

    return p_grad


def inner_or_not(input, pqrfile):
    centers, rs, _ = read_pqr(pqrfile, device="cpu", ratio=1)
    centers = centers.numpy()
    rs = rs.numpy()
    for j, cen in enumerate(centers):
        X_norm = np.linalg.norm(input - cen, ord=2, axis=1)

        if j == 0:
            index_inner = np.where(X_norm < rs[j])[0]
        else:
            index_inner = list(set(index_inner).union(set(list(np.where(X_norm < rs[j])[0]))))
    index_out = np.where(np.linalg.norm(input - centers[0], ord=2, axis=1) < 1e10)[0]
    index_out = list(set(index_out) - set(index_inner))

    return index_inner, index_out


def test_data_net(device, args, a):
    h = 0.025
    X = np.arange(0, 1 + h, h)
    Shape = X.shape
    Y = np.arange(0, 1 + h, h)
    Z = np.arange(0, 1 + h, h)
    X, Y, Z = np.meshgrid(X, Y, Z)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = Z.reshape(-1, 1)

    input = np.hstack((X, Y, Z))
    index_inner, index_out = inner_or_not(input, args.pqrfile)
    input = torch.tensor(input).float()
    test_inner = input[index_inner, :].to(device)
    label_inner = u(test_inner, "inner", a, device).clone().detach()

    test_out = input[index_out, :].to(device)
    label_out = u(test_out, "out", a, device).clone().detach()

    return test_out, label_out, test_inner, label_inner


def main(args):
    if torch.cuda.is_available and args.cuda:
        device = "cuda"
        print("cuda is avaliable")
    else:
        device = "cpu"

    a = torch.tensor(args.a).to(device)
    b = torch.tensor(args.b).to(device)

    ### test data
    test_out, label_out, test_inner, label_inner = test_data_net(device, args, args.a)

    ### train data
    data = Data(args.pqrfile, args.box, device, ratio=1)
    Nout = data.SampleFromOut(args.Ntrain_out)
    index = random.sample(range(0, Nout.size()[0]), args.train_out)
    out = Nout[index, :].T
    x_out, y_out, z_out, input_out = data_transform(out)

    Ninner = data.SampleFromInner(args.Ntrain_inner)
    index = random.sample(range(0, Ninner.size()[0]), args.train_inner)
    inner = Ninner[index, :].T
    x_in, y_in, z_in, input_in = data_transform(inner)

    Gamma, F_direction = data.SampleFromGamma(args.Ntrain_gamma)
    index = random.sample(range(0, Gamma.size()[0]), args.train_gamma)
    gamma = Gamma[index, :].T
    f_direction = F_direction[index, :]
    x_gamma, y_gamma, z_gamma, input_gamma = data_transform(gamma)

    Nout_b = data.SampleFromBoundary(args.Ntrain_boundary)
    index = random.sample(range(0, Nout_b.size()[0]), args.train_boundary)
    out_b = Nout_b[index, :]
    out_b_label = u(out_b, "out", args.a, device)

    g_D = u(input_gamma, "out", a, device) - u(input_gamma, "inner", a, device)
    g_N = phi_grad(input_gamma, f_direction, a, device)
    z = torch.ones(g_D.size()).to(device)

    print("input_gamma ", input_gamma.size())
    print("input_in ", input_in.size())
    print("input_out ", input_out.size())
    print("out_b", out_b.size())

    net_inner = DeepRitzNet(m=args.unit).to(device)
    net_out = DeepRitzNet(m=args.unit).to(device)
    optimizer = optim.Adam(itertools.chain(net_inner.parameters(), net_out.parameters()), lr=args.lr)
    result = []
    t0 = time.time()
    task = {}
    task_loss = {}
    loss_history = []

    Traing_Mse_min = 1e10
    Traing_Mse_min_epoch = 0
    if not os.path.isdir("./outputs/" + args.filename + "/model"):
        os.makedirs("./outputs/" + args.filename + "/model")

    for epoch in range(args.nepochs):
        optimizer.zero_grad()
        U1 = net_inner(input_in)
        U_1x, U_1y, U_1z = gradient(U1, x_in, y_in, z_in)
        U_1xx = grad(U_1x, x_in)
        U_1yy = grad(U_1y, y_in)
        U_1zz = grad(U_1z, z_in)
        loss_in = torch.mean(
            (-a[0] * (U_1xx + U_1yy + U_1zz) + b[0] * U1 - f_grad(input_in, "inner", a, b, device)) ** 2
        )

        U1_b = net_inner(input_gamma)
        U2_b_in = net_out(input_gamma)
        loss_gammad = torch.mean((-U1_b + U2_b_in - g_D) ** 2)

        dU1_N = torch.autograd.grad(U1_b, input_gamma, grad_outputs=z, create_graph=True)[0]
        U1_N = (dU1_N * f_direction).sum(dim=1).view(-1, 1) * a[0]
        dU2_N = torch.autograd.grad(U2_b_in, input_gamma, grad_outputs=z, create_graph=True)[0]
        U2_N = (dU2_N * f_direction).sum(dim=1).view(-1, 1) * a[1]
        loss_gamman = torch.mean((-g_N - U1_N + U2_N) ** 2)

        U2 = net_out(input_out)
        U_2x, U_2y, U_2z = gradient(U2, x_out, y_out, z_out)
        U_2xx = grad(U_2x, x_out)
        U_2yy = grad(U_2y, y_out)
        U_2zz = grad(U_2z, z_out)
        loss_out = torch.mean(
            (-a[1] * (U_2xx + U_2yy + U_2zz) + b[1] * U2 - f_grad(input_out, "out", a, b, device)) ** 2
        )
        loss_boundary = torch.mean((net_out(out_b) - out_b_label) ** 2)

        ###########  INN
        NN = 3
        s = random.sample(range(1, NN), 1)[0]
        task["bd_add_bn"] = s / NN * loss_gammad / loss_gammad.data + (1 - s / NN) * loss_gamman / loss_gamman.data
        task["loss_in_add_out"] = loss_in + loss_out
        task["outb"] = loss_boundary
        task_loss["1"] = loss_in.item()
        task_loss["2"] = loss_gammad.item()
        task_loss["3"] = loss_gamman.item()
        task_loss["4"] = loss_out.item()
        task_loss["5"] = loss_boundary.item()

        scale = MGDA_train(epoch, task, task_loss, net_inner, net_out, optimizer, device, s, NN)

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
        index = random.sample(range(0, Nout.size()[0]), args.train_out)
        out = Nout[index, :].T
        x_out, y_out, z_out, input_out = data_transform(out)

        index = random.sample(range(0, Ninner.size()[0]), args.train_inner)
        inner = Ninner[index, :].T
        x_in, y_in, z_in, input_in = data_transform(inner)

        if not os.path.isdir("./outputs/" + args.filename + "/model"):
            os.makedirs("./outputs/" + args.filename + "/model")
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
    parser.add_argument("--pqrfile", type=str, default="transfer_two.pqr")
    parser.add_argument("--filename", type=str, default="rand")
    parser.add_argument("--train_inner", type=int, default=50)
    parser.add_argument("--train_gamma", type=int, default=100)
    parser.add_argument("--train_out", type=int, default=1000)
    parser.add_argument("--train_boundary", type=int, default=300)
    parser.add_argument("--print_num", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", type=str, default=True)
    parser.add_argument("--a", type=list, default=[1, 100])
    parser.add_argument("--b", type=list, default=[1, 100])
    parser.add_argument("--box", type=list, default=[0, 1, 0, 1, 0, 1])
    parser.add_argument("--change_epoch", type=int, default=2000)
    parser.add_argument("--unit", type=int, default=160)
    parser.add_argument("--Ntrain_inner", type=int, default=500)
    parser.add_argument("--Ntrain_gamma", type=int, default=100)
    parser.add_argument("--Ntrain_out", type=int, default=10000)
    parser.add_argument("--Ntrain_boundary", type=int, default=300)
    parser.add_argument("--save", type=str, default=False)
    args = parser.parse_args()
    main(args)
