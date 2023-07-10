"""This is directly copied from https://github.com/bzlu-Group/INN/blob/main/INN_E4/GenerateData.py
"""

import torch

pi = 3.141592653


class Data(object):
    def __init__(self, r0, L, box, device):
        self.r0 = r0
        self.L = torch.tensor(L).to(device)
        self.box = torch.tensor(box).to(device)
        self.device = device

    def SampleFromGamma(self, num):
        theta = torch.rand(num, device=self.device).view(-1, 1)  # [0,pi]
        psi = torch.rand(num, device=self.device).view(-1, 1)  # [0,2pi]
        x = self.L[0] + self.r0 * torch.sin(theta * pi) * torch.cos(psi * 2 * pi)
        y = self.L[1] + self.r0 * torch.sin(theta * pi) * torch.sin(psi * 2 * pi)
        z = self.L[2] + self.r0 * torch.cos(theta * pi)
        X = torch.cat((x, y, z), dim=1)

        return X

    def SampleFromInner(self, num):
        L = self.L
        r = self.r0 * torch.rand(num, device=self.device).view(-1, 1)
        theta = torch.rand(num, device=self.device).view(-1, 1)
        psi = torch.rand(num, device=self.device).view(-1, 1)
        x = self.L[0] + r * torch.sin(theta * pi) * torch.cos(psi * 2 * pi)
        y = self.L[1] + r * torch.sin(theta * pi) * torch.sin(psi * 2 * pi)
        z = self.L[2] + r * torch.cos(theta * pi)
        X = torch.cat((x, y, z), dim=1)

        return X

    def SampleFromOut(self, num):
        X = self.__sampleFromDomain(2 * num)
        y = torch.norm(X - self.L, dim=1)
        location = torch.where(y > self.r0)[0]
        X = X[location, :]
        X = X[0:num, :]

        return X

    def __sampleFromDomain(self, num):
        xmin, xmax, ymin, ymax, zmin, zmax = self.box
        x = torch.rand(num, device=self.device).view(-1, 1) * (xmax - xmin) + xmin
        y = torch.rand(num, device=self.device).view(-1, 1) * (ymax - ymin) + ymin
        z = torch.rand(num, device=self.device).view(-1, 1) * (zmax - zmin) + zmin
        X = torch.cat((x, y, z), dim=1)

        return X

    def SampleFromBoundary(self, num):
        xmin, xmax, ymin, ymax, zmin, zmax = self.box
        n = int(num / 6)

        rand0 = torch.rand(n).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        b = torch.ones_like(rand0).to(self.device) * zmin
        P = torch.cat((rand0, rand1, b), dim=1)

        rand0 = torch.rand(n).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        b = torch.ones_like(rand0).to(self.device) * zmax
        P = torch.cat((P, torch.cat((rand0, rand1, b), dim=1)), dim=0)

        rand0 = torch.rand(n).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0).to(self.device) * ymin
        P = torch.cat((P, torch.cat((rand0, b, rand1), dim=1)), dim=0)

        rand0 = torch.rand(n).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0).to(self.device) * ymax
        P = torch.cat((P, torch.cat((rand0, b, rand1), dim=1)), dim=0)

        rand0 = torch.rand(n).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        rand1 = torch.rand(n).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0).to(self.device) * xmin
        P = torch.cat((P, torch.cat((b, rand0, rand1), dim=1)), dim=0)

        rand0 = torch.rand(n).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        rand1 = torch.rand(n).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0).to(self.device) * xmax
        P = torch.cat((P, torch.cat((b, rand0, rand1), dim=1)), dim=0)

        return P.to(self.device)
