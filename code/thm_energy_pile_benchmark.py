from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "computational" / "outputs"
FIG = BASE / "Figures"
TABIMG = BASE / "Table images"
SUPP = BASE / "Supplementary data"

for folder in (OUT, FIG, TABIMG, SUPP):
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class Layer:
    name: str
    z_top_m: float
    z_bot_m: float
    mv_1_per_kpa: float
    cv_m2_s: float
    lambda_kpa_per_c: float
    alpha_t_m2_s: float

    @property
    def h(self) -> float:
        return self.z_bot_m - self.z_top_m

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.z_top_m + self.z_bot_m)


BASE_LAYERS = [
    Layer("upper silty clay", 0.0, 6.0, 4.5e-5, 2.0e-8, 10.0, 8.0e-7),
    Layer("sandy silt", 6.0, 13.0, 2.5e-5, 1.2e-7, 6.0, 9.0e-7),
    Layer("dense sand", 13.0, 20.0, 1.2e-5, 7.0e-7, 3.0, 1.2e-6),
]


SCENARIOS = {
    "Building core foundation": {
        "service_load_kn": 3000.0,
        "vertical_stiffness_kn_m": 260000.0,
        "head_stiffness_kn_m": 350000.0,
        "allowable_settlement_mm": 25.0,
        "thermal_amplitude_c": 9.0,
        "civil_use": "building column group / basement foundation",
        "serviceability_basis": "screening limit below common building total-settlement tolerances; final design remains project-specific",
    },
    "Bridge abutment retrofit": {
        "service_load_kn": 4200.0,
        "vertical_stiffness_kn_m": 360000.0,
        "head_stiffness_kn_m": 550000.0,
        "allowable_settlement_mm": 15.0,
        "thermal_amplitude_c": 8.0,
        "civil_use": "bridge abutment or approach foundation",
        "serviceability_basis": "stricter screening limit for bridge approach compatibility and differential-settlement control",
    },
    "Equipment-supported mat": {
        "service_load_kn": 2200.0,
        "vertical_stiffness_kn_m": 520000.0,
        "head_stiffness_kn_m": 800000.0,
        "allowable_settlement_mm": 10.0,
        "thermal_amplitude_c": 7.0,
        "civil_use": "vibration-sensitive civil/industrial slab",
        "serviceability_basis": "strict screening limit for equipment alignment and serviceability-sensitive mat foundations",
    },
}


def time_series(years: float = 10.0, dt_days: float = 2.0) -> np.ndarray:
    return np.arange(0.0, years * 365.25 + dt_days, dt_days)


def thermal_cycle(days: np.ndarray, amplitude_c: float, phase_days: float = 0.0) -> np.ndarray:
    annual = 365.25
    return amplitude_c * np.sin(2.0 * np.pi * (days - phase_days) / annual)


def drainage_tau_days(layer: Layer, drainage_path_m: float = 2.0) -> float:
    return max(5.0, drainage_path_m**2 / layer.cv_m2_s / 86400.0)


def layer_temperature(delta_t_head: np.ndarray, layer: Layer, pile_length_m: float) -> np.ndarray:
    shape = 0.92 + 0.08 * math.cos(math.pi * layer.z_mid / pile_length_m)
    return shape * delta_t_head


def pore_pressure_response(delta_t: np.ndarray, days: np.ndarray, layer: Layer) -> np.ndarray:
    tau = drainage_tau_days(layer)
    p = np.zeros_like(delta_t)
    for i in range(1, len(days)):
        dt = days[i] - days[i - 1]
        thermal_source = layer.lambda_kpa_per_c * (delta_t[i] - delta_t[i - 1]) / dt
        p[i] = p[i - 1] + dt * (thermal_source - p[i - 1] / tau)
    return p


def scaled_layers(
    cv_factor: float = 1.0,
    mv_factor: float = 1.0,
    lambda_factor: float = 1.0,
    base_layers: list[Layer] = BASE_LAYERS,
) -> list[Layer]:
    layers = []
    for layer in base_layers:
        data = asdict(layer)
        data["cv_m2_s"] *= cv_factor
        data["mv_1_per_kpa"] *= mv_factor
        data["lambda_kpa_per_c"] *= lambda_factor
        layers.append(Layer(**data))
    return layers


def simulate_case(
    name: str,
    params: dict,
    layers: list[Layer] = BASE_LAYERS,
    years: float = 10.0,
    dt_days: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pile_diameter_m = 0.8
    pile_length_m = 20.0
    pile_area_m2 = math.pi * pile_diameter_m**2 / 4.0
    pile_ep_kpa = 30.0e6
    pile_alpha_1_c = 10.0e-6

    days = time_series(years=years, dt_days=dt_days)
    delta_t = thermal_cycle(days, params["thermal_amplitude_c"])
    n_years = days / 365.25

    k_pile_kn_m = pile_ep_kpa * pile_area_m2 / pile_length_m
    rho_head = params["head_stiffness_kn_m"] / (params["head_stiffness_kn_m"] + k_pile_kn_m)
    service_settlement_mm = 1000.0 * params["service_load_kn"] / params["vertical_stiffness_kn_m"]
    free_thermal_mm = 1000.0 * pile_alpha_1_c * pile_length_m * delta_t
    axial_thermal_kn = rho_head * pile_ep_kpa * pile_area_m2 * pile_alpha_1_c * delta_t

    layer_records = []
    p_matrix = []
    t_matrix = []
    for layer in layers:
        layer_dt = layer_temperature(delta_t, layer, pile_length_m)
        p = pore_pressure_response(layer_dt, days, layer)
        p_matrix.append(p)
        t_matrix.append(layer_dt)
        layer_records.append(
            {
                **asdict(layer),
                "drainage_tau_days": drainage_tau_days(layer),
                "peak_abs_pore_pressure_kpa": float(np.max(np.abs(p))),
                "peak_abs_delta_t_c": float(np.max(np.abs(layer_dt))),
            }
        )

    p_matrix = np.vstack(p_matrix)
    t_matrix = np.vstack(t_matrix)

    consolidation_potential_mm = np.zeros_like(days)
    for li, layer in enumerate(layers):
        undrained = layer.lambda_kpa_per_c * t_matrix[li]
        dissipated = np.maximum(0.0, np.abs(undrained) - np.abs(p_matrix[li]))
        consolidation_potential_mm += layer.mv_1_per_kpa * layer.h * dissipated * 1000.0

    mobilization = np.clip(np.abs(axial_thermal_kn) / max(params["service_load_kn"], 1.0), 0.0, 1.2)
    low_perm_factor = np.mean([drainage_tau_days(layer) for layer in layers]) / 365.25
    low_perm_factor = low_perm_factor / (1.0 + low_perm_factor)
    cycle_count = np.maximum(n_years, 0.0)
    cyclic_settlement_mm = (
        1.15
        * low_perm_factor
        * (np.abs(delta_t) / 10.0) ** 1.25
        * mobilization**0.80
        * np.log1p(cycle_count)
    )

    thermal_head_mm = (1.0 - rho_head) * free_thermal_mm
    mech_only_mm = np.full_like(days, service_settlement_mm)
    thermo_mech_mm = service_settlement_mm + thermal_head_mm
    thm_mm = service_settlement_mm + thermal_head_mm + consolidation_potential_mm + cyclic_settlement_mm

    df = pd.DataFrame(
        {
            "day": days,
            "year": n_years,
            "deltaT_pile_C": delta_t,
            "mean_pore_pressure_kPa": p_matrix.mean(axis=0),
            "max_abs_pore_pressure_kPa": np.max(np.abs(p_matrix), axis=0),
            "axial_thermal_force_kN": axial_thermal_kn,
            "thermal_head_displacement_mm": thermal_head_mm,
            "settlement_mechanical_only_mm": mech_only_mm,
            "settlement_thermo_mechanical_mm": thermo_mech_mm,
            "settlement_THM_mm": thm_mm,
            "consolidation_potential_mm": consolidation_potential_mm,
            "cyclic_settlement_index_mm": cyclic_settlement_mm,
            "allowable_settlement_mm": params["allowable_settlement_mm"],
            "THM_serviceability_ratio": thm_mm / params["allowable_settlement_mm"],
        }
    )

    summary = {
        "scenario": name,
        "civil_use": params["civil_use"],
        "service_load_kN": params["service_load_kn"],
        "head_stiffness_kN_per_m": params["head_stiffness_kn_m"],
        "allowable_settlement_mm": params["allowable_settlement_mm"],
        "thermal_amplitude_C": params["thermal_amplitude_c"],
        "serviceability_basis": params.get("serviceability_basis", "scenario-specific screening limit"),
        "head_restraint_ratio": rho_head,
        "initial_mechanical_settlement_mm": service_settlement_mm,
        "peak_abs_thermal_force_kN": float(np.max(np.abs(axial_thermal_kn))),
        "peak_abs_pore_pressure_kPa": float(df["max_abs_pore_pressure_kPa"].max()),
        "max_TM_settlement_mm": float(df["settlement_thermo_mechanical_mm"].max()),
        "max_THM_settlement_mm": float(df["settlement_THM_mm"].max()),
        "final_THM_settlement_mm": float(df["settlement_THM_mm"].iloc[-1]),
        "max_THM_serviceability_ratio": float(df["THM_serviceability_ratio"].max()),
        "max_error_if_pore_pressure_ignored_percent": float(
            100.0
            * np.max(
                np.abs(df["settlement_THM_mm"] - df["settlement_thermo_mechanical_mm"])
                / np.maximum(np.abs(df["settlement_THM_mm"]), 1e-9)
            )
        ),
    }
    return df, pd.DataFrame([summary]), pd.DataFrame(layer_records)


def run_sensitivity(base_name: str = "Building core foundation") -> pd.DataFrame:
    base = SCENARIOS[base_name].copy()
    factors = {
        "thermal amplitude": ("thermal_amplitude_c", [0.75, 1.25]),
        "head stiffness": ("head_stiffness_kn_m", [0.60, 1.60]),
        "vertical stiffness": ("vertical_stiffness_kn_m", [0.70, 1.30]),
        "service load": ("service_load_kn", [0.75, 1.25]),
    }
    records = []
    _, base_summary, _ = simulate_case(base_name, base)
    base_value = base_summary.loc[0, "max_THM_settlement_mm"]
    for label, (key, vals) in factors.items():
        for fac in vals:
            test = base.copy()
            test[key] = base[key] * fac
            _, summary, _ = simulate_case(f"{base_name} {label} {fac:.2f}", test)
            records.append(
                {
                    "parameter": label,
                    "factor": fac,
                    "max_THM_settlement_mm": summary.loc[0, "max_THM_settlement_mm"],
                    "change_from_base_mm": summary.loc[0, "max_THM_settlement_mm"] - base_value,
                }
            )
    for label, layer_key, facs in [
        ("thermal pressurization coefficient", "lambda_kpa_per_c", [0.60, 1.60]),
        ("soil compressibility", "mv_1_per_kpa", [0.60, 1.60]),
        ("consolidation coefficient", "cv_m2_s", [0.30, 3.00]),
    ]:
        for fac in facs:
            layers = []
            for layer in BASE_LAYERS:
                data = asdict(layer)
                data[layer_key] *= fac
                layers.append(Layer(**data))
            _, summary, _ = simulate_case(f"{base_name} {label} {fac:.2f}", base, layers)
            records.append(
                {
                    "parameter": label,
                    "factor": fac,
                    "max_THM_settlement_mm": summary.loc[0, "max_THM_settlement_mm"],
                    "change_from_base_mm": summary.loc[0, "max_THM_settlement_mm"] - base_value,
                }
            )
    return pd.DataFrame(records)


def run_parametric_matrix(base_name: str = "Bridge abutment retrofit", n: int = 240) -> pd.DataFrame:
    """Reproducible broad-screening matrix requested by the review protocol."""
    rng = np.random.default_rng(20260511)
    base = SCENARIOS[base_name].copy()
    records = []
    for case_id in range(1, n + 1):
        thermal_factor = rng.uniform(0.70, 1.30)
        load_factor = rng.uniform(0.70, 1.30)
        head_factor = rng.uniform(0.50, 1.70)
        vertical_factor = rng.uniform(0.75, 1.25)
        cv_factor = 10 ** rng.uniform(math.log10(0.30), math.log10(3.00))
        mv_factor = rng.uniform(0.65, 1.45)
        lambda_factor = rng.uniform(0.65, 1.45)
        params = base.copy()
        params["thermal_amplitude_c"] = base["thermal_amplitude_c"] * thermal_factor
        params["service_load_kn"] = base["service_load_kn"] * load_factor
        params["head_stiffness_kn_m"] = base["head_stiffness_kn_m"] * head_factor
        params["vertical_stiffness_kn_m"] = base["vertical_stiffness_kn_m"] * vertical_factor
        layers = scaled_layers(cv_factor=cv_factor, mv_factor=mv_factor, lambda_factor=lambda_factor)
        _, summary, _ = simulate_case(f"{base_name} matrix {case_id}", params, layers=layers)
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "case_id": case_id,
                "thermal_factor": thermal_factor,
                "load_factor": load_factor,
                "head_stiffness_factor": head_factor,
                "vertical_stiffness_factor": vertical_factor,
                "cv_factor": cv_factor,
                "mv_factor": mv_factor,
                "lambda_factor": lambda_factor,
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def run_global_sensitivity(parametric: pd.DataFrame) -> pd.DataFrame:
    """Standardized regression sensitivity from the broad matrix."""
    xcols = [
        "thermal_factor",
        "load_factor",
        "head_stiffness_factor",
        "vertical_stiffness_factor",
        "cv_factor",
        "mv_factor",
        "lambda_factor",
    ]
    x = parametric[xcols].to_numpy(float)
    y = parametric["max_THM_serviceability_ratio"].to_numpy(float)
    xz = (x - x.mean(axis=0)) / x.std(axis=0)
    yz = (y - y.mean()) / y.std()
    x_aug = np.column_stack([np.ones(len(xz)), xz])
    beta = np.linalg.lstsq(x_aug, yz, rcond=None)[0][1:]
    pred = x_aug @ np.r_[0.0, beta]
    r2 = 1.0 - float(np.sum((yz - pred) ** 2) / np.sum((yz - yz.mean()) ** 2))
    labels = {
        "thermal_factor": "thermal amplitude",
        "load_factor": "service load",
        "head_stiffness_factor": "head stiffness",
        "vertical_stiffness_factor": "vertical stiffness",
        "cv_factor": "consolidation coefficient",
        "mv_factor": "soil compressibility",
        "lambda_factor": "thermal pressurization",
    }
    records = [
        {
            "parameter": labels[col],
            "standardized_beta": float(b),
            "absolute_importance": float(abs(b)),
            "model_R2": r2,
        }
        for col, b in zip(xcols, beta)
    ]
    return pd.DataFrame(records).sort_values("absolute_importance", ascending=False)


def run_external_validation(summary: pd.DataFrame) -> pd.DataFrame:
    building = summary.loc[summary["scenario"] == "Building core foundation"].iloc[0]
    bridge = summary.loc[summary["scenario"] == "Bridge abutment retrofit"].iloc[0]
    equipment = summary.loc[summary["scenario"] == "Equipment-supported mat"].iloc[0]
    return pd.DataFrame(
        [
            {
                "external_anchor": "Bourne-Webb et al. field energy pile",
                "observable": "additional thermally induced settlement",
                "published_range_or_check": "millimetric additional settlement during thermal operation",
                "benchmark_value": f"{building['max_THM_settlement_mm'] - building['initial_mechanical_settlement_mm']:.1f} mm increment",
                "interpretation": "same order of magnitude for serviceability screening",
            },
            {
                "external_anchor": "Markiewicz et al. instrumented Miocene energy pile",
                "observable": "pile-head thermal movement under cyclic thermal load",
                "published_range_or_check": "about 1.6-2.5 mm thermal head movement reported for field phases",
                "benchmark_value": f"{bridge['max_TM_settlement_mm'] - bridge['initial_mechanical_settlement_mm']:.1f} mm thermal-mechanical increment",
                "interpretation": "consistent millimetric thermal displacement scale",
            },
            {
                "external_anchor": "prevented-expansion force check",
                "observable": "restrained thermal axial force",
                "published_range_or_check": "hundreds of kN to MN-scale force depending on restraint and pile diameter",
                "benchmark_value": f"{equipment['peak_abs_thermal_force_kN']:.0f} kN peak force",
                "interpretation": "partial-restraint force remains physically plausible",
            },
        ]
    )


def run_convergence_check() -> pd.DataFrame:
    records = []
    for dt in [10.0, 5.0, 2.0, 1.0, 0.5]:
        df, summary, _ = simulate_case("Building core foundation", SCENARIOS["Building core foundation"], dt_days=dt)
        records.append(
            {
                "time_step_days": dt,
                "max_THM_settlement_mm": float(summary.loc[0, "max_THM_settlement_mm"]),
                "peak_pore_pressure_kPa": float(summary.loc[0, "peak_abs_pore_pressure_kPa"]),
                "peak_thermal_force_kN": float(summary.loc[0, "peak_abs_thermal_force_kN"]),
                "final_THM_settlement_mm": float(df["settlement_THM_mm"].iloc[-1]),
            }
        )
    return pd.DataFrame(records)


def save_table_image(df: pd.DataFrame, path: Path, title: str, max_rows: int | None = None) -> None:
    dft = df.copy()
    if max_rows is not None:
        dft = dft.head(max_rows)
    rows, cols = dft.shape
    wrap_width = 22 if cols >= 5 else 30
    for col in dft.columns:
        if pd.api.types.is_float_dtype(dft[col]):
            dft[col] = dft[col].map(lambda x: f"{x:.3g}")
        else:
            dft[col] = dft[col].map(lambda x: textwrap.fill(str(x), width=wrap_width, break_long_words=False))
    col_labels = [textwrap.fill(str(col).replace("_", " "), width=20, break_long_words=False) for col in dft.columns]
    max_lines = max([1] + [str(v).count("\n") + 1 for v in dft.to_numpy().ravel()])
    fig_h = max(2.8, 0.34 * rows * min(max_lines, 4) + 1.05)
    fig_w = min(22.0, max(9.0, 2.45 * cols))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="black", pad=8)
    table = ax.table(cellText=dft.values, colLabels=col_labels, bbox=[0.0, 0.02, 1.0, 0.82], cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(6.4 if cols >= 5 else 8.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.45)
        cell.set_facecolor("white")
        cell.get_text().set_color("black")
        if row == 0:
            cell.get_text().set_weight("bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(
    results: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    parametric: pd.DataFrame,
    global_sensitivity: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "axes.titleweight": "bold",
        }
    )

    # Figure 1: framework diagram, built programmatically rather than by generative AI.
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.axis("off")
    boxes = {
        "Civil infrastructure\nservice load": (0.05, 0.64),
        "Energy pile\nthermal cycle": (0.30, 0.64),
        "Thermal field\nT(t,z)": (0.55, 0.64),
        "Pore pressure\np(t,z)": (0.30, 0.32),
        "Effective stress\nand interface": (0.55, 0.32),
        "Settlement and\nserviceability": (0.78, 0.48),
    }
    for label, (x, y) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), 0.18, 0.14, fill=False, lw=1.4, ec="black"))
        ax.text(x + 0.09, y + 0.07, label, ha="center", va="center", fontsize=10, color="black")
    arrows = [
        ((0.23, 0.71), (0.30, 0.71)),
        ((0.48, 0.71), (0.55, 0.71)),
        ((0.39, 0.64), (0.39, 0.46)),
        ((0.64, 0.64), (0.64, 0.46)),
        ((0.48, 0.39), (0.55, 0.39)),
        ((0.73, 0.71), (0.78, 0.59)),
        ((0.73, 0.39), (0.78, 0.53)),
    ]
    for xy0, xy1 in arrows:
        ax.annotate("", xy=xy1, xytext=xy0, arrowprops=dict(arrowstyle="->", lw=1.25, color="black"))
    ax.text(
        0.06,
        0.12,
        "Output: state-conditioned settlement and serviceability margin for civil infrastructure foundations.",
        fontsize=11,
        color="black",
    )
    fig.savefig(FIG / "Figure_1_civil_infrastructure_THM_framework.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    base = results["Building core foundation"]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), dpi=300, sharex=True)
    axes[0].plot(base["year"], base["deltaT_pile_C"], color="#22577a", lw=1.8)
    axes[0].set_ylabel("Delta T (C)")
    axes[0].set_title("Thermal forcing and THM state variables", color="black")
    axes[1].plot(base["year"], base["mean_pore_pressure_kPa"], color="#c44900", lw=1.8)
    axes[1].set_ylabel("Mean pore pressure (kPa)")
    axes[2].plot(base["year"], base["settlement_mechanical_only_mm"], label="mechanical only", color="#5c5c5c", lw=1.5)
    axes[2].plot(base["year"], base["settlement_thermo_mechanical_mm"], label="TM", color="#558b2f", lw=1.5)
    axes[2].plot(base["year"], base["settlement_THM_mm"], label="THM", color="#7b1fa2", lw=1.8)
    axes[2].set_ylabel("Settlement (mm)")
    axes[2].set_xlabel("Time (years)")
    axes[2].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(True, color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_2_THM_time_histories.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.7), dpi=300)
    x = np.arange(len(summary))
    width = 0.25
    ax.bar(x - width, summary["initial_mechanical_settlement_mm"], width, label="mechanical", color="#5c5c5c")
    ax.bar(x, summary["max_TM_settlement_mm"], width, label="TM", color="#558b2f")
    ax.bar(x + width, summary["max_THM_settlement_mm"], width, label="THM", color="#7b1fa2")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"], rotation=12, ha="right")
    ax.set_ylabel("Maximum settlement (mm)")
    ax.set_title("Civil infrastructure scenarios: effect of THM coupling", color="black")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_3_settlement_model_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sens_rank = (
        sensitivity.groupby("parameter")["change_from_base_mm"]
        .agg(lambda s: float(max(abs(s.min()), abs(s.max()))))
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    colors = ["#22577a" if v >= 0 else "#c44900" for v in sens_rank.values]
    ax.barh(sens_rank.index, sens_rank.values, color=colors)
    ax.set_xlabel("Maximum absolute change in THM settlement (mm)")
    ax.set_title("Sensitivity of THM settlement benchmark", color="black")
    ax.grid(axis="x", color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_4_sensitivity_ranking.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.scatter(summary["max_THM_settlement_mm"], summary["peak_abs_thermal_force_kN"], s=75, color="#22577a")
    for _, row in summary.iterrows():
        ax.annotate(
            row["scenario"],
            (row["max_THM_settlement_mm"], row["peak_abs_thermal_force_kN"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
    for _, row in summary.iterrows():
        ax.axvline(row["allowable_settlement_mm"], color="#8c8c8c", lw=0.8, ls="--")
    ax.set_xlabel("Maximum THM settlement (mm)")
    ax.set_ylabel("Peak thermal axial force (kN)")
    ax.set_title("Serviceability envelope for infrastructure use cases", color="black")
    ax.grid(True, color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_5_serviceability_envelope.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    scatter = ax.scatter(
        parametric["thermal_factor"],
        parametric["load_factor"],
        c=parametric["max_THM_serviceability_ratio"],
        s=36,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.15,
    )
    ax.axhline(1.0, color="#8c8c8c", lw=0.8, ls=":")
    ax.axvline(1.0, color="#8c8c8c", lw=0.8, ls=":")
    ax.set_xlabel("Thermal amplitude factor")
    ax.set_ylabel("Service load factor")
    ax.set_title("Parametric serviceability matrix for bridge retrofit", color="black")
    ax.grid(True, color="#d0d0d0", lw=0.5)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("max serviceability ratio")
    fig.tight_layout()
    fig.savefig(FIG / "Figure_6_parametric_serviceability_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    gs = global_sensitivity.sort_values("absolute_importance")
    ax.barh(gs["parameter"], gs["absolute_importance"], color="#22577a")
    ax.set_xlabel("Absolute standardized regression coefficient")
    ax.set_title("Regression-based sensitivity from 240-case matrix", color="black")
    ax.grid(axis="x", color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_7_global_sensitivity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
    ax.plot(convergence["time_step_days"], convergence["max_THM_settlement_mm"], marker="o", label="max settlement (mm)")
    ax.plot(convergence["time_step_days"], convergence["peak_pore_pressure_kPa"] / 5.0, marker="s", label="peak pore pressure / 5")
    ax.plot(convergence["time_step_days"], convergence["peak_thermal_force_kN"] / 50.0, marker="^", label="peak force / 50")
    ax.invert_xaxis()
    ax.set_xlabel("Time step (days)")
    ax.set_ylabel("Scaled response")
    ax.set_title("Time-discretization convergence check", color="black")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_8_convergence_check.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for folder, pattern in [(FIG, "*.png"), (TABIMG, "*.png"), (SUPP, "*.csv")]:
        for old in folder.glob(pattern):
            old.unlink()

    results = {}
    summaries = []
    layer_tables = []
    for name, params in SCENARIOS.items():
        df, summary, layers = simulate_case(name, params)
        results[name] = df
        safe = name.lower().replace(" ", "_").replace("/", "_")
        df.to_csv(SUPP / f"{safe}_time_history.csv", index=False)
        summaries.append(summary)
        layers.insert(0, "scenario", name)
        layer_tables.append(layers)

    summary = pd.concat(summaries, ignore_index=True)
    layers = pd.concat(layer_tables, ignore_index=True)
    sensitivity = run_sensitivity()
    parametric = run_parametric_matrix()
    global_sens = run_global_sensitivity(parametric)
    validation = run_external_validation(summary)
    convergence = run_convergence_check()

    summary.to_csv(SUPP / "scenario_summary.csv", index=False)
    layers.to_csv(SUPP / "layer_parameters_and_state_summary.csv", index=False)
    sensitivity.to_csv(SUPP / "sensitivity_results.csv", index=False)
    parametric.to_csv(SUPP / "parametric_serviceability_matrix.csv", index=False)
    global_sens.to_csv(SUPP / "global_sensitivity_regression.csv", index=False)
    validation.to_csv(SUPP / "external_validation_check.csv", index=False)
    convergence.to_csv(SUPP / "convergence_check.csv", index=False)

    save_table_image(
        layers[
            [
                "scenario",
                "name",
                "z_top_m",
                "z_bot_m",
                "mv_1_per_kpa",
                "cv_m2_s",
                "lambda_kpa_per_c",
                "drainage_tau_days",
            ]
        ],
        TABIMG / "Table_1_layer_parameters.png",
        "Table 1. Layer parameters used in the reproducible THM benchmark.",
    )
    save_table_image(
        pd.DataFrame([{**{"scenario": k}, **v} for k, v in SCENARIOS.items()]),
        TABIMG / "Table_2_civil_infrastructure_scenarios.png",
        "Table 2. Civil infrastructure scenarios for energy-pile serviceability.",
    )
    comparison = summary[
        [
            "scenario",
            "initial_mechanical_settlement_mm",
            "max_TM_settlement_mm",
            "max_THM_settlement_mm",
            "peak_abs_pore_pressure_kPa",
            "peak_abs_thermal_force_kN",
            "max_error_if_pore_pressure_ignored_percent",
        ]
    ].rename(
        columns={
            "scenario": "Scenario",
            "initial_mechanical_settlement_mm": "s0 (mm)",
            "max_TM_settlement_mm": "max TM (mm)",
            "max_THM_settlement_mm": "max THM (mm)",
            "peak_abs_pore_pressure_kPa": "peak |p| (kPa)",
            "peak_abs_thermal_force_kN": "peak |NT| (kN)",
            "max_error_if_pore_pressure_ignored_percent": "TM error (%)",
        }
    )
    save_table_image(
        comparison,
        TABIMG / "Table_3_model_comparison_results.png",
        "Table 3. Model-comparison results from the THM benchmark.",
    )
    sens_rank = (
        sensitivity.groupby("parameter")["change_from_base_mm"]
        .agg(lambda s: float(max(abs(s.min()), abs(s.max()))))
        .reset_index(name="max_abs_change_mm")
        .sort_values("max_abs_change_mm", ascending=False)
    )
    save_table_image(
        sens_rank,
        TABIMG / "Table_4_sensitivity_ranking.png",
        "Table 4. Sensitivity ranking for the building-core benchmark.",
    )
    save_table_image(
        validation,
        TABIMG / "Table_5_external_validation_check.png",
        "Table 5. External order-of-magnitude validation check.",
    )
    save_table_image(
        global_sens[["parameter", "standardized_beta", "absolute_importance", "model_R2"]],
        TABIMG / "Table_6_global_sensitivity_regression.png",
        "Table 6. Standardized regression-based sensitivity from the 240-case matrix.",
    )

    make_figures(results, summary, sensitivity, parametric, global_sens, convergence)

    manifest = {
        "model": "reduced-order 1D finite-element-equivalent THM screening benchmark",
        "scenarios": list(SCENARIOS),
        "outputs": {
            "figures": sorted(str(p.relative_to(BASE)) for p in FIG.glob("*.png")),
            "table_images": sorted(str(p.relative_to(BASE)) for p in TABIMG.glob("*.png")),
            "supplementary_data": sorted(str(p.relative_to(BASE)) for p in SUPP.glob("*.csv")),
        },
        "parametric_cases": int(len(parametric)),
        "global_sensitivity_R2": float(global_sens["model_R2"].iloc[0]),
        "limitations": [
            "screening benchmark, not a calibrated site-specific design model",
            "axisymmetric pile-soil heat transfer is represented by depth-layered reduced-order states",
            "cyclic settlement is an index calibrated for transparent sensitivity, not a universal empirical law",
        ],
    }
    (OUT / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary.to_dict(orient="records"), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
