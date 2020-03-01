
try:
    import math
    import numpy as np
    import time
    t1 = time.time_ns()
    # import xlwt
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as e:
    input(e)


class function():

    min_Grad = 0.0001

    def __init__(self, f, **kwargs):

        self.func = f
        self.solve = kwargs['solve']  # [Array(Tupel,),(,)] Given input data and the should output
        self.search = kwargs['search']  # number of variables unknown
        self.coeff = np.zeros(shape=(self.search))

        if 'init_val' in kwargs:
            self.coeff = kwargs['init_val']

    def find_recursive(self, coeff, mask, acc, mzr):

        capt = False
        cf = coeff
        gradients = []
        add = 1

        # Get y_0 in order to determine the gradient
        # coefficients should be last effective values above and the newly determined lying beneath
        if (mask + 1) < len(cf):
            cf = self.find_recursive(cf, mask + 1, acc, mzr)

        y_0 = self.abs_Dev(cf)
        #if mask == 0: print('[{0}, {1}] {2}'.format(str(cf[0]), str(cf[1]), str(y_0)))

        while True:

            cf[mask] += add

            # If not last coefficient
            # Call recursive solving
            if (mask + 1) < len(cf):
                cf = self.find_recursive(cf, mask + 1, acc, mzr)

            y_1 = self.abs_Dev(cf)

            # If gradient pulls towards minimum
            if (y_1 - y_0) <= 0:
                pass

            else:
                capt = True
                # Repair last step
                #cf[mask] -= add
                # Shorten add over negative minimizer
                add /= -mzr


            # Append new gradient: Δy / Δx
            gradients.append((y_1 - y_0) / add)
            y_0 = y_1
            # Managing Gradient record
            while len(gradients) > 2:
                gradients.pop(0)

            # If gradients are not changing fast enlarge add
            # To save computing time
            if self.constant(gradients) and not capt:
                add *= mzr

            # Acceptable accuracy?
            if abs(add) < acc:
                return cf


    # Determine whether gradients are being similarly
    def constant(self, arr):
        if len(arr) >= 2:
            if abs(arr[0] - arr[1]) < self.min_Grad:
                return True
        return False

    # Return the error of calculated Deviation based on given solve data
    def error(self, coeff):
        #       Total absolute deviation * number of given datapoints
        #       -----------------------------------------------------
        #       sum of given datapoint-outcome (last value of 'point')

        return self.abs_Dev(coeff) * len(self.solve) / (sum([point[len(point) - 1] for point in self.solve]))

    # return Absolute Deviation of function on given coefficients and (Multidimensional) Points
    def abs_Dev(self, coeff):
        sum = 0
        for point in self.solve:
            sum += abs(self.func(point[0], *coeff) - point[1])

        return sum

    def print_Wb(self, name, **kwargs):

        wb = xlwt.Workbook()
        ws = wb.add_sheet(name[1])

        runden = 2
        if 'runden' in kwargs:
            runden = kwargs['runden']

        for x in range(0, 101):
            for y in range(0, 101):

                if x == 0 and y == 0: continue

                r_1 = x * 8000 / 101 + 0
                c = y * 300 / 101 + 0

                if x == 0:
                    ws.write(x, y, str(round(c, runden)) + 'µF')
                    continue

                if y == 0:
                    ws.write(x, y, str(round(r_1, runden)) + 'Ω')
                    continue

                ws.write(x, y, round(self.error([r_1, c]) * 100, runden))

        wb.save(name[0])


R_1_index = 0
R_2 = 1200
c_index = 1
s = 40
runden = 2

U0 = 300


def _Uc(vm, *args):
    # Uc * e ^(-(40m/v) / (c * R))
    return (args[R_1_index] * U0 / (args[R_1_index] + R_2)) * math.exp(-(s / vm) / (args[R_1_index] * args[c_index] / 1000000))


def frange(*args):
    if len(args) == 1:
        start = 0
        stop = args[0]
    elif len(args) == 2:
        start = args[0]
        stop = args[1]

    schritt = 1
    if len(args) == 3:
        schritt = args[3]

    while start < stop:
        yield start
        start += schritt

    pplt.contourplot()


if __name__ == '__main__':

    Kat_Nr = 26

    vm = 75 + Kat_Nr / math.sqrt(3)

    # Testdata
    v = [(vm * 5 / 6, vm * 5 / 6),
         (vm * 8 / 9, vm * 8 / 9),
         (vm, vm),
         (vm * 9 / 8, vm * 9 / 8),
         (vm * 6 / 5, vm * 6 / 5)
         ]

    # v = [i for i in range(75, 110)]
    # for i in range(len(v)):
    #     v[i] = (v[i], v[i])

    f = function(_Uc, solve=v, search=2, init_val=[1200, 100])
    f.coeff = f.find_recursive(f.coeff, 0, 0.0001, 2)

    print('Für Nummer {0} liegt das Minimum der Abweichungen für R2 = 1200Ω und U0 = {4}V bei:\n'
          'R1 = {1}Ω\nc = {2}µF\nSumme der Abweichungen = {3}%\n\n'
          'Berücksichtigte Geschwindigkeits/Spannungs-punkte:\n{5}'.format(
           Kat_Nr,
           round(f.coeff[R_1_index], runden),
           round(f.coeff[c_index], runden),
           round(f.error(f.coeff) * 100, runden),
           round(U0, runden),
           [str(round(point[len(point) - 1], runden)) for point in f.solve]))

    t_ges = time.time_ns() - t1

    print('Process finished in ' + str(t_ges /1000000) + 'ms')
    input()
    ###############################################
    # PLOTTING
    ###############################################


    # c_val = range(1, 1000, 1)       # Startvalue Maxvalue Stepsize of plot
    # r_val = range(1, 10000, 10)
    #
    # c_mesh, r_mesh = np.meshgrid(c_val, r_val)
    # z_mesh = []
    #
    # for x in range(len(r_val)):
    #     row = []
    #     for y in range(len(c_val)):
    #         #print(str(r_mesh[x][y]) + str(c_mesh[x][y]))
    #         row.append(f.error([r_mesh[x][y], c_mesh[x][y]]))
    #     z_mesh.append(row)
    #



    # fig, ax = plt.subplots()
    # #CP = ax.contour(r_mesh, c_mesh, z_mesh, [x/1000 for x in range(0,100)])
    # #ax.clabel(CP, inline=1)
    # LP = ax.plot(x, y)
    # plt.ylim((0, 3))
    # plt.ylabel('Σ Fehler [%]')
    # plt.xlabel('Katalognummer')
    # fig.savefig('Lineplot.png')
    # plt.show()

