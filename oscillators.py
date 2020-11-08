import numpy as np
import scipy
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation


def oscillatorN(E):
    [m, k, xi, vi, tf] = E

    def xprime(t, Y):
        L = []
        # edge case (i = 0)
        L.append(Y[1])
        L.append(-k[0] / m[0] * Y[0] + k[1] / m[0] * (Y[2] - Y[0]))
        for i in range(1, len(m) - 1):
            L.append(Y[2 * i + 1])
            L.append(-k[i] / m[i] * (Y[2 * i] - Y[2 * i - 2]) + k[i + 1] / m[i] * (Y[2 * i + 2] - Y[2 * i]))

        # edge case (i = len(m) - 1)
        L.append(Y[2 * (len(m) - 1) + 1])
        L.append(-k[len(m) - 1] / m[len(m) - 1] * (Y[2 * (len(m) - 1)] - Y[2 * (len(m) - 1) - 2]) + k[len(m)] / m[
            len(m) - 1] * (-Y[2 * (len(m) - 1)]))

        return L

    X = []
    for j in range(len(m)):
        X.append(xi[j])
        X.append(vi[j])

    solution = solve_ivp(xprime, (0, tf), X, dense_output=True)

    t = np.linspace(0, tf, math.floor(tf * 30))
    z = solution.sol(t)

    X = []  # This is the cleaned version of z, with only the positions and not the velocities

    for i in range(len(z) // 2):
        X.append(z[2 * i].T)

    XT = np.array(X).T
    fig = plt.figure()
    ax = plt.axes(xlim=(0, len(m)), ylim=(-1.5, 1.5))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    pos = []
    for i in range(len(m)):
        pos.append(i)

    def animate(i):
        line.set_data(pos, XT[4*i])
        return line,

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=len(XT)//4, interval=5, blit=True)
    plt.xlabel('oscillator number')
    plt.ylabel('position from equilibrium (m)')
    plt.title('Travelling wave (500 oscillators)')
    plt.show()


# oscillatorN([[1, 0.5], [1, 1, 0.8], [0, 0], [1, 2], 30])

def identical(N, tf):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N):
        m.append(1)
        k.append(1)
        vi.append(1)
        xi.append(0)
    k.append(1)
    return ([m, k, xi, vi, tf])


# oscillatorN(identical(50,200))

def wavewalls(N, q, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]

    for i in range(N):
        m.append(1)
        k.append(S)
        vi.append(0)
        xi.append(0)
    k.append(S)

    vi[q] = 1  # the qth mass moves initially, the others start at rest

    return ([m, k, xi, vi, tf])


#oscillatorN(wavewalls(1000, 500, 1000, 2))


def wavenowalls(N, q, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]

    for i in range(N):
        m.append(1)
        if i == 0:
            vi.append(0)  # All speeds 0 for now
            k.append(0)  # First spring is absent
        else:
            vi.append(0)
            k.append(S)
        xi.append(0)
    k.append(0)  # Last spring is absent

    vi[q] = 1  # the qth mass moving initially

    return ([m, k, xi, vi, tf])

#oscillatorN(wavenowalls(100,50,200,1))

#oscillatorN(wavenowalls(1000, 500, 1000, 4))

#travelling wave
def sinWave(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N//2):
        xi.append(math.sin(2*math.pi*i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-math.sqrt(S) * 2*math.pi / lmda * math.cos(2*math.pi*i / lmda))

    for i in range(N//2, N):
        xi.append(0)
        k.append(S)
        m.append(1)
        vi.append(0)

    #additional k
    k.append(S)

    return [m, k, xi, vi, tf]

oscillatorN(sinWave(50, 500, 250, 4))

#standing wave
def standingWave(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N//2):
        xi.append(math.sin(2*math.pi*i / lmda))
        k.append(S)
        m.append(1)
        vi.append(0)

    for i in range(N//2, N):
        xi.append(0)
        k.append(S)
        m.append(1)
        vi.append(0)

    #additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(standingWave(50, 1000, 250, 4))

#decreasing the spring constant
def sinNotWave1(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N // 2):
        xi.append(0.5 * math.sin(2 * math.pi * i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-0.5 * math.sqrt(S) * 2 * math.pi / lmda * math.cos(2 * math.pi * i / lmda))

    for i in range(N // 2, N):
        xi.append(0)
        k.append(S*(N-i)*2/N)
        m.append(1)
        vi.append(0)

    # additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(sinNotWave1(50, 500, 250, 4))

#decreasing the mass
def sinNotWave2(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N // 2):
        xi.append(0.5 * math.sin(2 * math.pi * i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-0.5 * math.sqrt(S) * 2 * math.pi / lmda * math.cos(2 * math.pi * i / lmda))

    for i in range(N // 2, N):
        xi.append(0)
        k.append(S)
        m.append((N-0.9*i)*2/N)
        vi.append(0)

    # additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(sinNotWave2(50, 500, 250, 4))

#increasing the mass
def sinNotWave3(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N // 2):
        xi.append(0.5 * math.sin(2 * math.pi * i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-0.5 * math.sqrt(S) * 2 * math.pi / lmda * math.cos(2 * math.pi * i / lmda))

    for i in range(N // 2, N):
        xi.append(0)
        k.append(S)
        m.append(10/N * (i-N/2)+1)
        vi.append(0)

    # additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(sinNotWave3(50, 500, 500, 4))

#sudden increase in spring constant
def sinNotWave4(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N // 2):
        xi.append(0.5 * math.sin(2 * math.pi * i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-0.5 * math.sqrt(S) * 2 * math.pi / lmda * math.cos(2 * math.pi * i / lmda))

    for i in range(N // 2, N):
        xi.append(0)
        k.append(3*S)
        m.append(1)
        vi.append(0)

    # additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(sinNotWave4(50, 500, 500, 4))

#sudden increase in mass
def sinNotWave5(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N // 2):
        xi.append(0.5 * math.sin(2 * math.pi * i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-0.5 * math.sqrt(S) * 2 * math.pi / lmda * math.cos(2 * math.pi * i / lmda))

    for i in range(N // 2, N):
        xi.append(0)
        k.append(S)
        m.append(3)
        vi.append(0)

    # additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(sinNotWave5(50, 500, 500, 4))

#negative spring constant
def sinNotWave6(lmda, N, tf, S):
    [m, k, xi, vi, tf] = [[], [], [], [], tf]
    for i in range(N // 2):
        xi.append(0.5 * math.sin(2 * math.pi * i / lmda))
        k.append(S)
        m.append(1)
        vi.append(-0.5 * math.sqrt(S) * 2 * math.pi / lmda * math.cos(2 * math.pi * i / lmda))

    for i in range(N // 2, N):
        xi.append(0)
        k.append(-S/100)
        m.append(3)
        vi.append(0)

    # additional k
    k.append(S)

    return [m, k, xi, vi, tf]

#oscillatorN(sinNotWave6(50, 500, 500, 4))
