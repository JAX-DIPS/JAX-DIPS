import torch
import random

pi = 3.141592653


def read_pqr(file_name, device, ratio):
    """
    input: pqr file
    return: centers,rs,qs  [datype: array]
    """
    f = open(file_name)
    line = f.readline()
    centers = []
    rs = []
    qs = []
    while line:
        centers.append(list(map(float, line.split()[5:8])))
        qs.append(list(map(float, line.split()[8:9]))[0])
        rs.append(list(map(float, line.split()[9:]))[0])
        line = f.readline()
    centers = torch.tensor(centers).to(device) / ratio
    rs = torch.tensor(rs).to(device) / ratio
    qs = torch.tensor(qs).to(device)

    return centers, rs, qs


class Data(object):
    def __init__(self, pqr_file, box, device, ratio):
        centers, rs, qs = read_pqr(pqr_file, device, ratio=ratio)
        self.rs = rs
        self.qs = qs
        self.centers = centers
        self.box = torch.tensor(box).to(device)
        self.device = device

    def __single_sphere_bound(self, num, L, r0):
        """
        L is the center of sphere
        output: sphere surface point  ;shape[number,3]
        """
        theta = torch.rand(num, device=self.device).view(-1, 1)  # [0,pi]
        psi = torch.rand(num, device=self.device).view(-1, 1)  # [0,2pi]
        x = L[0] + r0 * torch.sin(theta * pi) * torch.cos(psi * 2 * pi)
        y = L[1] + r0 * torch.sin(theta * pi) * torch.sin(psi * 2 * pi)
        z = L[2] + r0 * torch.cos(theta * pi)
        X = torch.cat((x, y, z), dim=1)

        return X

    def SampleFromGamma(self, totle_num):
        """
        num : sample num from each sphere
        """
        k = 1
        bound_sign = 0
        num = totle_num / len(self.rs)
        R_min = torch.sum(self.rs) / len(self.rs) / 10
        for i, center in enumerate(self.centers):
            # sphere area: 4*pi*r**2
            sphere = self.__single_sphere_bound(int(num * (self.rs[i] / R_min) ** 2), center, self.rs[i])
            for j, cen in enumerate(self.centers):
                if i == j:
                    pass
                else:
                    if torch.norm(center - cen) > self.rs[i] + self.rs[j]:
                        pass
                    elif sphere.size()[0] == 0:
                        k = 0
                        pass
                    else:
                        X_norm = torch.norm(sphere - cen, dim=1)
                        index = torch.where(X_norm > self.rs[j])
                        sphere = sphere[index[0], :]
                        k = 1
            if k == 0:
                pass
            elif i == 0 or bound_sign == 0:
                bound = sphere
                f_direction = (bound - center) / self.rs[i]
                bound_sign = 1.0
            else:
                bound = torch.cat((bound, sphere), dim=0)
                f_direction = torch.cat((f_direction, (sphere - center) / self.rs[i]), dim=0)
        try:
            index = random.sample(range(0, bound.shape[0]), totle_num)
        except:
            print("bound data is less than espected!", bound.shape[0], totle_num)
            return bound, f_direction

        bound = bound[index, :]
        f_direction = f_direction[index, :]

        return bound, f_direction

    def __single_sphere_inner(self, num, L, r0):
        """
        Generate inner and out data and avoid the interface!!!
        return inner point of the sphere; shape[number,3]
        """
        if num < 6:
            num = 5
        r = (torch.rand(num, device=self.device) * r0).view(-1, 1)
        theta = torch.rand(len(r), device=self.device).view(-1, 1)  # [0,pi]
        psi = torch.rand(len(r), device=self.device).view(-1, 1)  # [0,2pi]
        x = L[0] + r * torch.sin(theta * pi) * torch.cos(psi * 2 * pi)
        y = L[1] + r * torch.sin(theta * pi) * torch.sin(psi * 2 * pi)
        z = L[2] + r * torch.cos(theta * pi)
        X = torch.cat((x, y, z), dim=1)

        return X

    def SampleFromInner(self, totle_num):
        """
        sample point inner
        """
        num = int(totle_num / len(self.rs))
        R_min = torch.sum(self.rs) / len(self.rs) / 2
        for i, center in enumerate(self.centers):
            # sphere v 4/3*pi*r**3
            if i == 0:
                inner = self.__single_sphere_inner(int(num * (self.rs[i] / R_min) ** 3), center, self.rs[i])
            else:
                sphere = self.__single_sphere_inner(int(num * (self.rs[i] / R_min) ** 3), center, self.rs[i])
                for j, cen in enumerate(self.centers[:i]):
                    X_norm = torch.norm(sphere - cen, dim=1)
                    index = torch.where(X_norm > self.rs[j])
                    sphere = sphere[index[0], :]
                inner = torch.cat((inner, sphere), dim=0)

        index = random.sample(range(0, inner.shape[0]), totle_num)
        inner = inner[index, :]

        return inner.to(self.device)

    def SampleFromOut(self, num):
        """
        num is the the sample num
        return point [-2.5,2.5]^3 split two shpere ,
        sphere's center is c1 and c2,raduis is r
        """
        X = self.__sampleFromDomain(2 * num)

        for i, center in enumerate(self.centers):
            y = torch.norm(X - center, dim=1)
            location = torch.where(y > self.rs[i])[0]
            X = X[location, :]

        try:
            X = X[0:num, :]
            return X
        except:
            self.SampleFromOut(2 * num)

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

        rand0 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        b = torch.ones_like(rand0, device=self.device).to(self.device) * zmin
        array = torch.cat((rand0, rand1, b), dim=1)

        rand0 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        b = torch.ones_like(rand0, device=self.device).to(self.device) * zmax
        array = torch.cat((array, torch.cat((rand0, rand1, b), dim=1)), dim=0)

        rand0 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0, device=self.device).to(self.device) * ymin
        array = torch.cat((array, torch.cat((rand0, b, rand1), dim=1)), dim=0)

        rand0 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (xmax - xmin) + xmin
        rand1 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0, device=self.device).to(self.device) * ymax
        array = torch.cat((array, torch.cat((rand0, b, rand1), dim=1)), dim=0)

        rand0 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        rand1 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0, device=self.device).to(self.device) * xmin
        array = torch.cat((array, torch.cat((b, rand0, rand1), dim=1)), dim=0)

        rand0 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (ymax - ymin) + ymin
        rand1 = torch.rand(n, device=self.device).view(-1, 1).to(self.device) * (zmax - zmin) + zmin
        b = torch.ones_like(rand0, device=self.device).to(self.device) * xmax
        array = torch.cat((array, torch.cat((b, rand0, rand1), dim=1)), dim=0)

        return array
