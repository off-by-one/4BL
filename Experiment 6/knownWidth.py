import csv
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt


X = []
I = []

guess=True

wave = 0.00067

with open('Aperture.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        a, b = float(row[0]), float(row[1])
        X.append(a)
        I.append(-b)

angles = np.array(X)
intens = np.array(I)

#print angles 
#print intens 


def func(t, d, b, A, p):
    return A * np.square( (np.cos (np.pi * d * np.cos(t + p) / wave )) * 
                          np.sinc(b * np.sin(t + p) / wave) )

two_slit = lambda t, d, A, p : A * np.square(np.cos (2 * np.pi * d * np.cos(t + p) / wave ))

one_slit = lambda t, b, A, p, off : A*np.sinc(b * (t + p) / wave) + off

p0=[ 0.00412771, 0.01161464,-1.49476485, 0.44144418]
#p0=[-0.44, -1.5, -0.02, 0.4]
#p0=[34702913e+00,  2.12904764e-01,  9.85808654e+02,  3.16523212e-04]

if guess:
    ts_pfit, ts_pcov = curve_fit(one_slit, angles, intens, p0=p0, maxfev=1000000)
else:
    ts_pfit, ts_pcov = curve_fit(one_slit, angles, intens, maxfev=1000000)
#fits, covs = curve_fit(func, angles, intens)

print ts_pfit
print np.sqrt(np.diag(ts_pcov))

# we'll use this to plot our first estimate. This might already be good enough for you
if guess:
    data_first_guess = one_slit(angles, *p0)

# recreate the fitted curve using the optimized parameters
data_fit = one_slit(angles, *ts_pfit)


plt.plot(angles, intens)
plt.plot(angles, data_fit, label='after fitting')
if guess:
    plt.plot(angles, data_first_guess, label='first guess')
plt.legend()
plt.show()

