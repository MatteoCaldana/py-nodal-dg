import matplotlib.pyplot as plt

import pyndg.viscosity_model
import pyndg.scalar_2d

if __name__ == "__main__":
    ev = pyndg.viscosity_model.EV2D(c_E=3.0, c_max=1.5)
    p = pyndg.scalar_2d.get_problem(ev, "RotatingWave", 1, 100, 0.025, None, None)

    while not p.done:
        print(p.params.final_time, "/", p.time.item())
        p.step()

    x = p.mesh.x.numpy().flatten()
    y = p.mesh.y.numpy().flatten()
    u = p.u.numpy().flatten()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.tricontourf(x, y, u, cmap="jet", vmin=u.min(), levels=30)
    plt.axis("equal")
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.show()
