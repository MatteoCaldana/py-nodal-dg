import pyndg.viscosity_model
import pyndg.euler_2d
import time

if __name__ == "__main__":
    t0 = time.time()
    ev = pyndg.viscosity_model.EV2D(c_max=0.5, c_E=1.0)
    p = pyndg.euler_2d.get_problem(ev, "Euler", 5, 32, 0.2, "RP12", None)

    print("N cells", p.mesh.x.shape, len(p.u.reshape((-1,))))

    while not p.done:
        print(f"{time.time() - t0:.2f} | {p.params.final_time} / {p.time.item():.4f}")
        p.step()

    import matplotlib.pyplot as plt

    x = p.mesh.x.flatten()
    y = p.mesh.y.flatten()
    rho = p.u[0].flatten()

    plt.figure()
    cs = plt.tricontourf(x, y, rho, cmap="jet", levels=30)
    plt.gcf().colorbar(cs)
    plt.ylim([-0.5, 0.5])
    plt.xlim([-0.5, 0.5])

    plt.figure()
    plt.tricontour(x, y, rho, levels=50, colors="black")
    plt.ylim([-0.5, 0.5])
    plt.xlim([-0.5, 0.5])

    plt.show()
