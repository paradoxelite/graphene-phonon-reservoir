"""Reduced-order, contact-aware simulation of coupled graphene drum modes.

This module is a numerical research model, not a fabricated-device digital twin.
It keeps one authoritative nominal air gap, models the electrostatic pressure as
voltage squared over the instantaneous series-capacitor separation, and exposes
only a transfer-matrix optical readout.  Entering the contact region raises an
exception instead of clipping the trajectory into an unmodelled regime.
"""

from __future__ import annotations

from numbers import Integral, Number, Real

import numpy as np

EPS0 = 8.8541878128e-12
EPS_OX = 3.9
N_SIO2 = 1.457 + 0j
N_SI = 3.881 + 0.019j
N_GRAPHENE = 2.6 + 1.3j
T_GRAPHENE = 0.335e-9


class SimulationDomainError(RuntimeError):
    """The numerical trajectory left the stated model domain."""


class ContactError(SimulationDomainError):
    """The membrane entered the unresolved contact region."""


def _require_integer(name: str, value, *, minimum: int) -> int:
    """Return an exact integer scalar while rejecting booleans and coercions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean or coerced value")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _require_real(name: str, value) -> float:
    """Return a finite real scalar without treating booleans as numbers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean or text")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _require_complex(name: str, value) -> complex:
    """Return a finite, non-zero numeric refractive index."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Number):
        raise TypeError(f"{name} must be a numeric scalar, not a boolean or text")
    normalized = complex(value)
    if not np.isfinite(normalized.real) or not np.isfinite(normalized.imag):
        raise ValueError(f"{name} must be finite")
    if normalized == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return normalized


def _require_real_array(name: str, value, *, nonempty: bool = True) -> np.ndarray:
    """Snapshot and validate a finite real array without provenance loss."""
    try:
        snapshot = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a regular real numeric array") from error
    for item in snapshot.flat:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise TypeError(
                f"{name} must be a real numeric array, not boolean, text, or complex"
            )
    try:
        values = snapshot.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a regular real numeric array") from error
    if nonempty and values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _reflectance_scalar(
    air_gap: float,
    *,
    wavelength: float,
    oxide: float,
    n_graphene: complex,
    n_oxide: complex,
    n_substrate: complex,
) -> float:
    if not np.isfinite(air_gap) or air_gap <= 0.0:
        raise ValueError("air_gap must be finite and positive")
    if wavelength <= 0.0 or oxide < 0.0:
        raise ValueError("wavelength must be positive and oxide non-negative")

    matrix = np.eye(2, dtype=complex)
    layers = (
        (n_graphene, T_GRAPHENE),
        (1.0 + 0j, air_gap),
        (n_oxide, oxide),
    )
    # With exp(-i*omega*t) and passive optical constants n + i*kappa,
    # forward propagation exp(i*k*z) attenuates.  This transfer direction
    # therefore uses -i in the characteristic matrix and agrees with the
    # independent recursive Fresnel construction exercised in the tests.
    for refractive_index, thickness in layers:
        phase = 2.0 * np.pi * refractive_index * thickness / wavelength
        cosine = np.cos(phase)
        sine = np.sin(phase)
        layer = np.array(
            [
                [cosine, -1j * sine / refractive_index],
                [-1j * refractive_index * sine, cosine],
            ],
            dtype=complex,
        )
        matrix = matrix @ layer

    b_term = matrix[0, 0] + matrix[0, 1] * n_substrate
    c_term = matrix[1, 0] + matrix[1, 1] * n_substrate
    reflection = (b_term - c_term) / (b_term + c_term)
    value = float(abs(reflection) ** 2)
    if not np.isfinite(value):
        raise SimulationDomainError("non-finite optical reflectance")
    return value


def tmm_reflectance(
    air_gap,
    *,
    wavelength: float = 632.8e-9,
    oxide: float = 285e-9,
    n_graphene: complex = N_GRAPHENE,
    n_oxide: complex = N_SIO2,
    n_substrate: complex = N_SI,
):
    """Normal-incidence reflectance of air/graphene/gap/SiO2/Si.

    ``air_gap`` may be a scalar or an array and is always the physical air gap;
    the oxide thickness is represented separately rather than folded into a
    second optical operating point.
    """

    values = _require_real_array("air_gap", air_gap)
    if np.any(values <= 0.0):
        raise ValueError("air_gap must be finite and positive")
    wavelength = _require_real("wavelength", wavelength)
    oxide = _require_real("oxide", oxide)
    n_graphene = _require_complex("n_graphene", n_graphene)
    n_oxide = _require_complex("n_oxide", n_oxide)
    n_substrate = _require_complex("n_substrate", n_substrate)
    flat = np.array(
        [
            _reflectance_scalar(
                float(gap),
                wavelength=wavelength,
                oxide=oxide,
                n_graphene=n_graphene,
                n_oxide=n_oxide,
                n_substrate=n_substrate,
            )
            for gap in values.ravel()
        ],
        dtype=float,
    ).reshape(values.shape)
    if values.ndim == 0:
        return float(flat)
    return flat


class GrapheneOscillatorReservoir:
    """Ring of coupled reduced-order drum modes with optical readout.

    The modes share one nominal cavity geometry.  Their frequencies and input
    electrode gains vary with a seeded draw.  ``nonlinear=False`` is a matched
    mechanical ablation: it preserves frequencies, masks, damping, coupling,
    electrostatics and optical readout while removing cubic stiffness and
    nonlinear damping.
    """

    def __init__(
        self,
        *,
        n: int = 24,
        f_lo: float = 10e6,
        f_hi: float = 40e6,
        Q: float = 20.0,
        m_eff: float = 2e-18,
        area: float = 1.77e-12,
        gap: float = 250e-9,
        oxide: float = 285e-9,
        V_dc: float = 1.5,
        V_ac: float = 0.15,
        duffing_onset: float = 5e-9,
        nonlinear_damping_ratio: float = 0.2,
        coupling_fraction: float = 0.03,
        wavelength: float = 632.8e-9,
        dt: float = 0.25e-9,
        steps_per_input: int = 8,
        input_dim: int = 1,
        seed: int = 0,
        nonlinear: bool = True,
        contact_margin_fraction: float = 0.05,
    ):
        n = _require_integer("n", n, minimum=2)
        input_dim = _require_integer("input_dim", input_dim, minimum=1)
        steps_per_input = _require_integer(
            "steps_per_input", steps_per_input, minimum=1
        )
        seed = _require_integer("seed", seed, minimum=0)
        if type(nonlinear) is not bool:
            raise TypeError("nonlinear must be an exact boolean")
        f_lo = _require_real("f_lo", f_lo)
        f_hi = _require_real("f_hi", f_hi)
        Q = _require_real("Q", Q)
        m_eff = _require_real("m_eff", m_eff)
        area = _require_real("area", area)
        gap = _require_real("gap", gap)
        oxide = _require_real("oxide", oxide)
        V_dc = _require_real("V_dc", V_dc)
        V_ac = _require_real("V_ac", V_ac)
        duffing_onset = _require_real("duffing_onset", duffing_onset)
        nonlinear_damping_ratio = _require_real(
            "nonlinear_damping_ratio", nonlinear_damping_ratio
        )
        coupling_fraction = _require_real("coupling_fraction", coupling_fraction)
        wavelength = _require_real("wavelength", wavelength)
        dt = _require_real("dt", dt)
        contact_margin_fraction = _require_real(
            "contact_margin_fraction", contact_margin_fraction
        )
        positive = {
            "f_lo": f_lo,
            "f_hi": f_hi,
            "Q": Q,
            "m_eff": m_eff,
            "area": area,
            "gap": gap,
            "duffing_onset": duffing_onset,
            "wavelength": wavelength,
            "dt": dt,
            "steps_per_input": steps_per_input,
        }
        if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError(
                "physical scales and integration settings must be positive"
            )
        if f_hi < f_lo or not np.isfinite(oxide) or oxide < 0.0:
            raise ValueError("invalid frequency interval or oxide thickness")
        nonnegative = {
            "coupling_fraction": coupling_fraction,
            "nonlinear_damping_ratio": nonlinear_damping_ratio,
        }
        if any(
            not np.isfinite(value) or value < 0.0 for value in nonnegative.values()
        ):
            raise ValueError("coupling and nonlinear damping must be finite and non-negative")
        if not np.isfinite(V_dc) or not np.isfinite(V_ac):
            raise ValueError("voltage scales must be finite")
        if not 0.0 < contact_margin_fraction < 1.0:
            raise ValueError("contact_margin_fraction must lie in (0, 1)")

        self.n = int(n)
        self.input_dim = int(input_dim)
        self.m = float(m_eff)
        self.area = float(area)
        self.gap = float(gap)
        self.oxide = float(oxide)
        self.V_dc = float(V_dc)
        self.V_ac = float(V_ac)
        self.Q = float(Q)
        self.wavelength = float(wavelength)
        self.dt = float(dt)
        self.steps_per_input = int(steps_per_input)
        self.nonlinear = bool(nonlinear)
        self.contact_margin_fraction = float(contact_margin_fraction)

        rng = np.random.default_rng(seed)
        self.f0 = rng.uniform(f_lo, f_hi, self.n)
        self.w0 = 2.0 * np.pi * self.f0
        self.gamma = self.w0 / self.Q
        self.win = rng.uniform(-1.0, 1.0, size=(self.n, self.input_dim))
        self.alpha = self.m * self.w0**2 / duffing_onset**2
        self.eta = nonlinear_damping_ratio * self.m * self.gamma / duffing_onset**2
        if not self.nonlinear:
            self.alpha = np.zeros_like(self.alpha)
            self.eta = np.zeros_like(self.eta)
        self.k_coupling = coupling_fraction * self.m * float(np.mean(self.w0**2))

        minimum_gap = self.contact_margin_fraction * self.gap
        maximum_gap = 1.5 * self.gap
        self._optical_gaps = np.linspace(minimum_gap, maximum_gap, 4096)
        self._optical_reflectance = tmm_reflectance(
            self._optical_gaps,
            wavelength=self.wavelength,
            oxide=self.oxide,
        )
        self._baseline_reflectance = float(
            tmm_reflectance(
                self.gap,
                wavelength=self.wavelength,
                oxide=self.oxide,
            )
        )
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.n

    @property
    def symbol_period(self) -> float:
        return self.dt * self.steps_per_input

    @property
    def effective_series_gap(self) -> float:
        return self.gap + self.oxide / EPS_OX

    def reset(self) -> None:
        self.z = np.zeros(self.n, dtype=float)
        self.v = np.zeros(self.n, dtype=float)
        self.minimum_air_gap = self.gap

    def _input_vector(self, u) -> np.ndarray:
        vector = np.atleast_1d(_require_real_array("input", u)).ravel()
        if vector.size != self.input_dim:
            raise ValueError(
                f"input must contain exactly {self.input_dim} finite value(s)"
            )
        if np.any(np.abs(vector) > 1.0):
            raise ValueError("normalized input must lie in [-1, 1]")
        return vector

    def _assert_model_domain(self, z: np.ndarray) -> None:
        physical_gap = self.gap - z
        if np.any(physical_gap <= self.contact_margin_fraction * self.gap):
            raise ContactError("trajectory entered the unresolved contact region")
        if np.any(physical_gap > self._optical_gaps[-1]):
            raise SimulationDomainError(
                "trajectory left the calibrated optical gap grid"
            )
        if not np.isfinite(z).all() or not np.isfinite(self.v).all():
            raise SimulationDomainError("non-finite mechanical state")

    def electrostatic_force(self, z, u) -> np.ndarray:
        displacement_values = _require_real_array("displacement", z)
        try:
            displacement = np.broadcast_to(displacement_values, (self.n,))
        except ValueError as error:
            raise ValueError(
                f"displacement must be scalar or broadcastable to {self.n} modes"
            ) from error
        self._assert_model_domain(displacement)
        voltage = self.V_dc + self.V_ac * (self.win @ self._input_vector(u))
        series_gap = self.gap - displacement + self.oxide / EPS_OX
        force = 0.5 * EPS0 * self.area * voltage**2 / series_gap**2
        if not np.isfinite(force).all():
            raise SimulationDomainError("non-finite electrostatic force")
        return force

    def _observe(self) -> np.ndarray:
        self._assert_model_domain(self.z)
        local_gap = self.gap - self.z
        reflectance = np.interp(
            local_gap,
            self._optical_gaps,
            self._optical_reflectance,
        )
        return reflectance - self._baseline_reflectance

    def step(self, u) -> np.ndarray:
        input_vector = self._input_vector(u)
        for _ in range(self.steps_per_input):
            self._assert_model_domain(self.z)
            coupling = self.k_coupling * (
                np.roll(self.z, 1) + np.roll(self.z, -1) - 2.0 * self.z
            )
            force = self.electrostatic_force(self.z, input_vector)
            acceleration = (
                -self.m * self.w0**2 * self.z
                - self.alpha * self.z**3
                + coupling
                - self.m * self.gamma * self.v
                - self.eta * self.z**2 * self.v
                + force
            ) / self.m
            self.v += self.dt * acceleration
            self.z += self.dt * self.v
            self.minimum_air_gap = min(
                self.minimum_air_gap,
                float(np.min(self.gap - self.z)),
            )
            self._assert_model_domain(self.z)
        return self._observe()

    def run(self, signal, *, washout: int = 0, reset: bool = True) -> np.ndarray:
        if type(reset) is not bool:
            raise TypeError("reset must be an exact boolean")
        values = _require_real_array("signal", signal, nonempty=False)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("signal shape does not match input_dim")
        washout = _require_integer("washout", washout, minimum=0)
        if washout > len(values):
            raise ValueError("washout must lie between zero and signal length")
        if reset:
            self.reset()
        states = np.empty((len(values), self.state_dim), dtype=float)
        for index, value in enumerate(values):
            states[index] = self.step(value)
        return states[washout:]

    def mechanical_energy(self) -> float:
        kinetic = 0.5 * self.m * float(np.sum(self.v**2))
        onsite = 0.5 * self.m * float(np.sum(self.w0**2 * self.z**2))
        cubic = 0.25 * float(np.sum(self.alpha * self.z**4))
        differences = self.z - np.roll(self.z, -1)
        coupling = 0.5 * self.k_coupling * float(np.sum(differences**2))
        return kinetic + onsite + cubic + coupling
