#!/usr/bin/env python3

import re
import numpy as np
from subprocess import Popen, PIPE
from itertools import product
from matplotlib import pyplot as plt


iteration = 0

def parse(name, text):
    mean, std = re.search(name + r"\s*=\s*(\S*)\s*\+/-\s*(\S*)", text).groups()
    return (float(mean), float(std))

def calculate(prefix, mode, Exc, Eyc, H, Ex, Ey, omega, phi, T, n, dt, alltime, tau):
    global iteration
    iteration += 1
    print("%d" % iteration)
    args = "%s %s %e %e %e %e %e %e %e %e %d %e %e tau %e\n" % (prefix, mode, Exc, Eyc, H, Ex, Ey, omega, phi, T, n, dt, alltime, tau)
    out, err = Popen(["./bilayer_graphene"], shell=True, stdin=PIPE, stdout=PIPE).communicate(input=args.encode("utf8"))
    data = out.decode("utf8")
    return {
        "v_x": parse("vx", data),
        "v_y": parse("vy", data),
        "power": parse("power", data)
    }

if __name__ == '__main__':
    Exc     = 0
    Eyc     = 0
    Hl      = np.linspace(0, 1000, 11)
    Ex      = 10
    Ey      = 0
    omega   = 5e11
    phi     = 0
    T       = 77
    n       = 500
    dt      = 1e-14
    alltime = 1e-9
    tau     = 3e-12
    # однозонное приближение
    one_band = [calculate("CVC1", "one_band", Exc, Eyc, H, Ex, Ey, omega, phi, T, n, dt, alltime, tau)["power"][0] for H in Hl]
    # двухзонное
    two_bands = [calculate("CVC2", "two_bands", Exc, Eyc, H, Ex, Ey, omega, phi, T, n, dt, alltime, tau)["power"][0] for H in Hl]
    print(one_band)
    print(two_bands)
    plt.plot(Hl, one_band, label="one band")
    plt.plot(Hl, two_bands, label="two bands")
    plt.grid()
    plt.legend()
    plt.show()

#
# нужно сделать расчёт для одной частицы
# полный расчёт вдоль фазовой траектории
# с полным логом и анализом по времени
#