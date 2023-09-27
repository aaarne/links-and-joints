import numpy as np
import matplotlib.pyplot as plt
from ..planarrobots import RobotPlot
from allerlei import Timer

_animations = list()

def plot_pendulum_trajectory(pendulum, t, q, dq,
                             plot_orientation=True,
                             first_state=False,
                             energy=False,
                             phase_plots=True,
                             plot_fkin=False,
                             auto_corr=False,
                             streamer=None,
                             animation_destination=None,
                             q0=None,
                             labels=None,
                             show_endeff=True,
                             **kwargs):
    n = pendulum.dof
    if labels is None:
        qlabels = {i: f'$q_{i + 1}$' for i in range(n)}
        dqlabels = {i: f'$\\omega_{i+1}$' for i in range(n)}
    else:
        qlabels = {i: labels[i] for i in range(n)}
        dqlabels = {i: f"$\\dot{{{labels[i].strip('$')}}}$" for i in range(n)}

    all_labels = {**qlabels, **{i+n: v for i, v in dqlabels.items()}}

    figures = []
    f, axes = plt.subplots(3 if q0 else 2, 1)
    for i in range(n):
        axes[0].plot(t, q[:, i], label=qlabels[i])
    if q0:
        delta = q - pendulum.equilibrium
        for i in range(n):
            axes[1].plot(t, delta[:, i], label=f'$\\Delta q_{i+1}$')
    for i in range(n):
        axes[2 if q0 else 1].plot(t, dq[:, i], label=dqlabels[i])
    for ax in axes:
        ax.grid()
        ax.legend()
    f.tight_layout()
    figures.append(f)

    if auto_corr:
        f, ax = plt.subplots()
        from scipy.signal import correlate
        if q0:
            delta = q - pendulum.equilibrium
            for i in range(n):
                ax.plot(correlate(delta[:, i], delta[:, i]))
        else:
            for i in range(n):
                ax.plot(correlate(q[:, i], q[:, i]))

    if n == 2:
        if phase_plots:
            f, axes = plt.subplots(2, 2)
            traj = np.c_[q, dq]
            for ax, a, b in [
                (axes[0, 0], 0, 1),
                (axes[0, 1], 2, 3),
                (axes[1, 0], 0, 2),
                (axes[1, 1], 1, 3),
            ]:
                ax.plot(traj[:, a], traj[:, b])
                ax.set_xlabel(all_labels[a])
                ax.set_ylabel(all_labels[b])
                ax.grid()
            f.tight_layout()
            figures.append(f)

    fkin = pendulum.forward_kinematics(q)
    max_cart_distance = np.max(np.abs(fkin[:, 0:2]))
    if plot_fkin:
        f, axt = plt.subplots()
        handles = [
            axt.plot(t, fkin[:, 0], label='x'),
            axt.plot(t, fkin[:, 1], label='y'),
        ]
        if plot_orientation:
            axo = axt.twinx()
            handles.append(axo.plot(t, fkin[:, 2], '--', label='φ'))
        plt.legend(handles=[h[0] for h in handles])
        plt.grid()
        f.tight_layout()
        figures.append(f)


    kinfun = getattr(pendulum, 'kinetic_energy', None)
    if energy and callable(kinfun):
        f, ax = plt.subplots()
        K = pendulum.kinetic_energy(q, dq)
        V = pendulum.potential_energy(q)
        E = K + V
        ax.plot(t, K, label='K')
        ax.plot(t, V, label='V')
        ax.plot(t, E, label='E')
        ax.grid()
        ax.legend()
        f.tight_layout()
        figures.append(f)

    link_fkin = pendulum.forward_kinematics_for_each_link(q)

    if first_state:
        f, ax = plt.subplots()
        for i in range(pendulum.dof):
            ax.plot(link_fkin[:, i, 0], link_fkin[:, i, 1], alpha=.3)
        plot = RobotPlot(ax, f=f)
        plot.plot_robot(pendulum, q[0])
        figures.append(f)

    f, ax = plt.subplots()
    for i in range(pendulum.dof):
        ax.plot(link_fkin[:, i, 0], link_fkin[:, i, 1], alpha=.3)
    plot = RobotPlot(ax, f=f)
    anim = plot.animated_trajectory(pendulum, q, t=t, streamer=streamer, lim=max_cart_distance*1.5, show_endeffector=show_endeff)
    if animation_destination is not None:
        with Timer("Writing pendulum simulation gif"):
            anim.save(f"{animation_destination}.gif", writer='imagemagick', fps=30)
    _animations.append(anim)
    figures.append(f)

    return figures, anim


