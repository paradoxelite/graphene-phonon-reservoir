from collections import UserList, deque

import numpy as np
import pytest

from physical_model import (
    N_GRAPHENE,
    N_SI,
    N_SIO2,
    T_GRAPHENE,
    ContactError,
    GrapheneOscillatorReservoir,
    tmm_reflectance,
)


class StatefulArray:
    def __init__(self, first, second):
        self._responses = (first, second)
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        response = self._responses[min(self.calls, 1)]
        self.calls += 1
        return np.asarray(response, dtype=dtype)


def test_tmm_reflectance_is_bounded_and_half_wave_periodic():
    wavelength = 632.8e-9
    gaps = np.linspace(80e-9, 500e-9, 31)
    first = tmm_reflectance(gaps, wavelength=wavelength, oxide=285e-9)
    shifted = tmm_reflectance(
        gaps + wavelength / 2,
        wavelength=wavelength,
        oxide=285e-9,
    )

    assert np.isfinite(first).all()
    assert np.all((0.0 <= first) & (first <= 1.0))
    np.testing.assert_allclose(first, shifted, rtol=0.0, atol=2e-12)


def test_tmm_reduces_to_air_substrate_fresnel_limit():
    substrate_index = 1.5
    expected = ((1.0 - substrate_index) / (1.0 + substrate_index)) ** 2
    values = tmm_reflectance(
        np.array([1e-12, 100e-9, 400e-9]),
        oxide=0.0,
        n_graphene=1.0 + 0.0j,
        n_oxide=1.0 + 0.0j,
        n_substrate=substrate_index + 0.0j,
    )

    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-14)


def test_tmm_matches_passive_recursive_fresnel_convention():
    """The characteristic matrix must agree with passive n + i*kappa optics."""
    wavelength = 632.8e-9
    oxide = 285e-9
    gaps = np.array([80e-9, 250e-9, 375e-9])

    def recursive_reflectance(gap):
        indices = [1.0 + 0.0j, N_GRAPHENE, 1.0 + 0.0j, N_SIO2, N_SI]
        thicknesses = [np.inf, T_GRAPHENE, gap, oxide, np.inf]
        reflection = (indices[-2] - indices[-1]) / (indices[-2] + indices[-1])
        for layer in range(len(indices) - 3, -1, -1):
            phase = np.exp(
                2j
                * 2.0
                * np.pi
                * indices[layer + 1]
                * thicknesses[layer + 1]
                / wavelength
            )
            interface = (indices[layer] - indices[layer + 1]) / (
                indices[layer] + indices[layer + 1]
            )
            reflection = (interface + reflection * phase) / (
                1.0 + interface * reflection * phase
            )
        return float(abs(reflection) ** 2)

    expected = np.array([recursive_reflectance(gap) for gap in gaps])
    actual = tmm_reflectance(gaps, wavelength=wavelength, oxide=oxide)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-15)


@pytest.mark.parametrize(
    ("args", "kwargs", "parameter"),
    [
        ((np.array([True]),), {}, "air_gap"),
        ((np.array(["2.5e-7"]),), {}, "air_gap"),
        ((250e-9,), {"wavelength": True}, "wavelength"),
        ((250e-9,), {"oxide": np.bool_(False)}, "oxide"),
        ((250e-9,), {"n_graphene": True}, "n_graphene"),
        ((250e-9,), {"n_oxide": "1.457"}, "n_oxide"),
    ],
)
def test_tmm_rejects_boolean_or_coerced_public_inputs(args, kwargs, parameter):
    with pytest.raises(TypeError, match=parameter):
        tmm_reflectance(*args, **kwargs)


def test_tmm_rejects_empty_and_nonfinite_optical_inputs():
    with pytest.raises(ValueError, match="air_gap"):
        tmm_reflectance(np.array([], dtype=float))
    with pytest.raises(ValueError, match="n_substrate"):
        tmm_reflectance(250e-9, n_substrate=np.inf + 0j)


def test_tmm_accepts_valid_numpy_real_and_complex_scalars():
    value = tmm_reflectance(
        np.float32(250e-9),
        wavelength=np.float64(632.8e-9),
        oxide=np.float32(285e-9),
        n_graphene=np.complex128(N_GRAPHENE),
        n_oxide=np.complex128(N_SIO2),
        n_substrate=np.complex128(N_SI),
    )
    assert np.isfinite(value)


def test_electrostatic_force_rejects_boolean_displacement():
    model = GrapheneOscillatorReservoir(n=2, steps_per_input=1)
    with pytest.raises(TypeError, match="displacement"):
        model.electrostatic_force(np.array([False]), 0.0)


def test_invalid_coupling_and_nonlinear_damping_fail_closed():
    for kwargs in (
        {"coupling_fraction": -0.01},
        {"nonlinear_damping_ratio": -0.01},
        {"coupling_fraction": np.nan},
        {"nonlinear_damping_ratio": np.nan},
    ):
        with pytest.raises(ValueError):
            GrapheneOscillatorReservoir(n=2, **kwargs)


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("n", 2.5),
        ("n", True),
        ("n", "2"),
        ("n", np.float64(2.0)),
        ("input_dim", 1.5),
        ("steps_per_input", 1.5),
        ("seed", 1.5),
        ("seed", False),
    ),
)
def test_reservoir_rejects_non_integral_or_boolean_count_parameters(
    parameter, value
):
    with pytest.raises(TypeError, match=parameter):
        GrapheneOscillatorReservoir(n=2, **{parameter: value})


@pytest.mark.parametrize("value", ("false", 0, 1, np.bool_(True)))
def test_reservoir_requires_an_exact_boolean_nonlinear_flag(value):
    with pytest.raises(TypeError, match="nonlinear"):
        GrapheneOscillatorReservoir(n=2, nonlinear=value)


def test_reservoir_accepts_numpy_integer_scalars_without_coercing_floats():
    model = GrapheneOscillatorReservoir(
        n=np.int64(2),
        input_dim=np.int64(1),
        steps_per_input=np.int64(2),
        seed=np.int64(7),
    )

    assert model.n == 2
    assert model.input_dim == 1
    assert model.steps_per_input == 2


def test_boolean_arrays_are_not_accepted_as_numeric_drive_signals():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="boolean"):
        model._input_vector(np.array([True]))
    with pytest.raises(TypeError, match="boolean"):
        model.run(np.array([True, False]))


def test_input_vector_rejects_boolean_mixed_with_real_before_numpy_promotion():
    model = GrapheneOscillatorReservoir(n=2, input_dim=2, seed=12)

    with pytest.raises(TypeError, match="boolean"):
        model._input_vector([True, 0.25])


def test_input_vector_rejects_boolean_inside_generic_array_like():
    model = GrapheneOscillatorReservoir(n=2, input_dim=2, seed=12)

    with pytest.raises(TypeError, match="boolean"):
        model._input_vector(UserList([True, 0.25]))


def test_input_vector_accepts_real_generic_array_like():
    model = GrapheneOscillatorReservoir(n=2, input_dim=2, seed=12)

    vector = model._input_vector(UserList([0.0, 0.25]))

    np.testing.assert_array_equal(vector, [0.0, 0.25])


def test_input_vector_rejects_stateful_invalid_snapshot_without_retry():
    model = GrapheneOscillatorReservoir(n=2, input_dim=2, seed=12)
    signal = StatefulArray([True, 0.25], [0.0, 0.25])

    with pytest.raises(TypeError, match="boolean"):
        model._input_vector(signal)

    assert signal.calls == 1


def test_input_vector_materializes_stateful_array_like_once():
    model = GrapheneOscillatorReservoir(n=2, input_dim=2, seed=12)
    signal = StatefulArray([0.0, 0.25], [True, 0.25])

    vector = model._input_vector(signal)

    np.testing.assert_array_equal(vector, [0.0, 0.25])
    assert signal.calls == 1


def test_run_rejects_boolean_mixed_with_real_before_numpy_promotion():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="boolean"):
        model.run([True, 0.25])


def test_run_rejects_boolean_mixed_with_integer_in_generic_array_like():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="boolean"):
        model.run(deque([True, 1]))


def test_run_accepts_real_generic_array_like():
    model = GrapheneOscillatorReservoir(n=2, steps_per_input=1, seed=12)

    states = model.run(deque([0.0, 0.25]))

    assert states.shape == (2, model.state_dim)


def test_run_rejects_stateful_invalid_snapshot_without_retry():
    model = GrapheneOscillatorReservoir(n=2, steps_per_input=1, seed=12)
    signal = StatefulArray([True, 0.25], [0.0, 0.25])

    with pytest.raises(TypeError, match="boolean"):
        model.run(signal)

    assert signal.calls == 1


def test_run_materializes_stateful_array_like_once():
    model = GrapheneOscillatorReservoir(n=2, steps_per_input=1, seed=12)
    signal = StatefulArray([0.0, 0.25], [True, 0.25])

    states = model.run(signal)

    assert states.shape == (2, model.state_dim)
    assert signal.calls == 1


def test_input_vector_rejects_textual_numeric_signal():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="text"):
        model._input_vector(np.array(["0.25"]))


def test_input_vector_rejects_complex_signal():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="complex"):
        model._input_vector(np.array([0.25 + 0.5j]))


def test_run_rejects_textual_numeric_signal():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="text"):
        model.run(np.array(["0.25"]))


def test_run_rejects_complex_signal():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="complex"):
        model.run(np.array([0.25 + 0.5j]))


@pytest.mark.parametrize("reset", (1, "true", np.bool_(True)))
def test_run_requires_an_exact_boolean_reset_flag(reset):
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    with pytest.raises(TypeError, match="reset"):
        model.run(np.array([0.0, 0.1]), reset=reset)


@pytest.mark.parametrize(
    "parameter",
    (
        "f_lo",
        "oxide",
        "V_dc",
        "coupling_fraction",
        "contact_margin_fraction",
    ),
)
@pytest.mark.parametrize("value", (True, "1.0"))
def test_physical_real_parameters_reject_booleans_and_text(parameter, value):
    with pytest.raises(TypeError, match=parameter):
        GrapheneOscillatorReservoir(n=2, **{parameter: value})


def test_physical_real_parameters_accept_numpy_float_scalars():
    model = GrapheneOscillatorReservoir(
        n=2,
        f_lo=np.float64(10e6),
        f_hi=np.float64(20e6),
        Q=np.float64(20.0),
        coupling_fraction=np.float64(0.03),
    )

    assert model.Q == 20.0


def test_normalized_input_rejects_values_outside_unit_interval():
    model = GrapheneOscillatorReservoir(n=2, seed=12)

    model._input_vector(-1.0)
    model._input_vector(1.0)
    with pytest.raises(ValueError):
        model._input_vector(-1.000001)
    with pytest.raises(ValueError):
        model._input_vector(1.000001)


def test_electrostatic_force_increases_when_the_air_gap_closes():
    model = GrapheneOscillatorReservoir(n=2, seed=1)
    z = np.array([0.0, 0.1 * model.gap])

    force = model.electrostatic_force(z, u=0.0)

    assert force.shape == (2,)
    assert np.isfinite(force).all()
    assert force[1] > force[0] > 0.0


def test_contact_fails_closed_instead_of_clipping_the_state():
    model = GrapheneOscillatorReservoir(n=3, seed=2)
    model.z[1] = model.gap * (1.0 - 0.5 * model.contact_margin_fraction)

    with pytest.raises(ContactError):
        model.step(0.0)


def test_minimum_air_gap_is_tracked_and_reset():
    model = GrapheneOscillatorReservoir(n=4, seed=21)
    model.run(np.linspace(-1.0, 1.0, 20))

    assert model.contact_margin_fraction < model.minimum_air_gap / model.gap <= 1.0

    model.reset()
    assert model.minimum_air_gap == model.gap


def test_damped_unforced_mechanical_energy_decreases():
    model = GrapheneOscillatorReservoir(
        n=6,
        seed=3,
        V_dc=0.0,
        V_ac=0.0,
        Q=8.0,
        dt=0.25e-9,
        steps_per_input=4,
    )
    model.z[:] = np.linspace(-0.2e-9, 0.2e-9, model.n)
    initial = model.mechanical_energy()

    for _ in range(300):
        model.step(0.0)
    final = model.mechanical_energy()

    assert np.isfinite(final)
    assert 0.0 <= final < initial


def test_temporal_refinement_preserves_the_symbol_rate_response():
    common = {
        "n": 6,
        "seed": 4,
        "f_lo": 10e6,
        "f_hi": 15e6,
        "V_dc": 0.5,
        "V_ac": 0.02,
        "Q": 15.0,
    }
    coarse = GrapheneOscillatorReservoir(
        **common,
        dt=0.5e-9,
        steps_per_input=4,
    )
    fine = GrapheneOscillatorReservoir(
        **common,
        dt=0.25e-9,
        steps_per_input=8,
    )
    signal = np.sin(2 * np.pi * np.arange(80) / 19.0)

    coarse_states = coarse.run(signal)
    fine_states = fine.run(signal)
    scale = max(float(np.sqrt(np.mean(fine_states**2))), 1e-12)
    relative_rms = float(np.sqrt(np.mean((coarse_states - fine_states) ** 2)) / scale)

    assert relative_rms < 0.08


def test_seed_reproducibility_and_seed_variation():
    signal = np.linspace(-0.5, 0.5, 20)
    first = GrapheneOscillatorReservoir(n=5, seed=9).run(signal)
    repeated = GrapheneOscillatorReservoir(n=5, seed=9).run(signal)
    different = GrapheneOscillatorReservoir(n=5, seed=10).run(signal)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
