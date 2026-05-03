import matplotlib.pyplot as plt

import pyndg.euler_1d
import pyndg.visualize_results_1d
import pyndg.viscosity_model

if __name__ == "__main__":
    p = pyndg.euler_1d.get_problem(
        pyndg.viscosity_model.EV1D(c_E=1.0, c_max=0.25),
        None,
        5,
        50,
        0.3,
        "SodTube",
        None,
    )
    pyndg.visualize_results_1d.contourf([p], [830])
    plt.figure()
    plt.plot(p.mesh.x.numpy(), p.u[0].numpy(), "ko")
    plt.show()
    print(p.time)
