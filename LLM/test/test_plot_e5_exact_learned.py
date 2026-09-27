import numpy as np

from visual.plot_e5_exact_learned import (
    backed_up_advantages,
    local_deltas_from_backed_up,
    target_sum_along_guidance,
)


def test_reconstructs_budgeted_max_backup():
    # Root [d0, d1] receives a round-1 child after token 0.
    lam = 0.5
    children = {(0, 0): [1]}
    deltas = [np.array([1.0, 2.0]), np.array([4.0])]
    final = [np.array([3.0, 2.0]), np.array([4.0])]
    recovered = local_deltas_from_backed_up([x.tolist() for x in final], children, lam)
    np.testing.assert_allclose(recovered[0], deltas[0])
    np.testing.assert_allclose(recovered[1], deltas[1])
    s0 = backed_up_advantages(recovered, children, [0, 1], 0, lam)
    s1 = backed_up_advantages(recovered, children, [0, 1], 1, lam)
    np.testing.assert_allclose(s0[0], [2.0, 2.0])
    np.testing.assert_allclose(s1[0], final[0])


def test_target_return_follows_guidance_choice():
    children = {(0, 0): [1]}
    guidance = [np.array([2.0, 1.0]), np.array([3.0])]
    target_delta = [np.array([0.2, 0.4]), np.array([-0.5])]
    value, terminal = target_sum_along_guidance(
        root=0,
        children=children,
        rounds=[0, 1],
        guidance=guidance,
        target_deltas=target_delta,
        budget=1,
        lam=0.5,
    )
    assert terminal == 1
    assert value == 0.2 + 0.5 * -0.5
