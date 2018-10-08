import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


from matplotlib import rc, rcParams
rc("text", usetex=True)
# rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
# rcParams["font.family"] += ["serif"]


N1 = 1000
x = np.linspace(-1, 1, N1)
y1 = np.concatenate([np.sqrt(1-x**2), -np.sqrt(1-x**2)])
x1 = np.concatenate([np.linspace(-1, 1, N1), np.linspace(-1, 1, N1)])

N2 = 1000
x = np.linspace(-1, 1, N2)
y2 = np.concatenate([1-abs(x), -1+abs(x)])
x2 = np.concatenate([np.linspace(-1, 1, N2), np.linspace(-1, 1, N2)])

n1 = 250
angle1 = np.arctan(y1[n1]/x1[n1])
print(angle1)
angle_deg1 = angle1*180/np.pi
angle_deg1 = -72

e1_xpos = 1.2
e2_xpos = 1.5

annotation_font_size = 22
label_font_size = 24

width1 = np.sqrt((y1[n1]-e2_xpos)**2 + (x1[n1]-e2_xpos)**2)
breadth1 = np.sqrt((y1[n1]-e2_xpos)**2 + (x1[n1]-e2_xpos)**2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(x1, y1, label=r"$\mathrm{L}^2$")
ax1.fill(x1, y1, color="red", alpha=0.4)

e1 = patches.Ellipse((e1_xpos, e2_xpos), 1.0, 1.105*width1,
                     angle=angle_deg1, fill=False, color="orange", 
                     linewidth=1.5)#, label=r"$\beta_\mathrm{Ridge}$")
ax1.add_patch(e1)

ax1.annotate(r"$\beta_\mathrm{Ridge}$", (e1_xpos, e2_xpos), fontsize=annotation_font_size)

ax1.axhline(0, color="0")
ax1.axvline(0, color="0")
ax1.legend(fontsize=label_font_size)
ax1.set_xlabel(r"$\beta_0$", fontsize=label_font_size)
ax1.set_ylabel(r"$\beta_1$", fontsize=label_font_size)
ax1.axis("equal")
ax1.grid(True)


# plt.show()
fig1.savefig("../../fig/l2_norm.pdf")


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x2, y2, label=r"$\mathrm{L}^1$")
ax2.fill(x2, y2, color="red", alpha=0.4)

ax2.annotate(r"$\beta_\mathrm{Lasso}$", (e1_xpos, e2_xpos), fontsize=annotation_font_size)

e1 = patches.Ellipse((e1_xpos, e2_xpos), 1.0, 1.25*width1,
                     angle=angle_deg1, fill=False, color="orange", 
                     linewidth=1.5)#, label=r"$\beta_\mathrm{Lasso}$")
ax2.add_patch(e1)

ax2.axhline(0, color="0")
ax2.axvline(0, color="0")
ax2.set_xlabel(r"$\beta_0$", fontsize=label_font_size)
ax2.set_ylabel(r"$\beta_1$", fontsize=label_font_size)
ax2.legend(fontsize=label_font_size)
ax2.axis("equal")
ax2.grid(True)

fig2.savefig("../../fig/l1_norm.pdf")

plt.show()
