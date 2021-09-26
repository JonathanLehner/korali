import numpy as np
import matplotlib.pyplot as plt

root = "/scratch/snx3000/pweber/korali/testhalfDisk/_trainingResults/sample00003460/"

velData = np.loadtxt(root+"velocity_1.dat", skiprows=1)
plt.plot(velData[:,0], velData[:,2], label='x-coordinate')
plt.plot(velData[:,0], velData[:,3], label='y-coordinate')
plt.ylim([0.8,1.4])
plt.legend()
plt.show(block=False)

plt.plot(velData[:,0], velData[:,7], label='x-velocity')
plt.plot(velData[:,0], velData[:,8], label='y-velocity')
plt.ylim([-1,1])
plt.show()
