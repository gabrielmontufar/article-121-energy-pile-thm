from __future__ import annotations

import json
import math
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
    },
    "Bridge abutment retrofit": {
        "service_load_kn": 4200.0,
        "vertical_stiffness_kn_m": 360000.0,
        "head_stiffness_kn_m": 550000.0,
        "allowable_settlement_mm": 15.0,
        "thermal_amplitude_c": 8.0,
        "civil_use": "bridge abutment or approach foundation",
    },
    "Equipment-supported mat": {
        "service_load_kn": 2200.0,
        "vertical_stiffness_kn_m": 520000.0,
        "head_stiffness_kn_m": 800000.0,
        "allowable_settlement_mm": 10.0,
        "thermal_amplitude_c": 7.0,
        "civil_use": "vibration-sensitive civil/industrial slab",
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


def simulate_case(name: str, params: dict, layers: list[Layer] = BASE_LAYERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    pile_diameter_m = 0.8
    pile_length_m = 20.0
    pile_area_m2 = math.pi * pile_diameter_m**2 / 4.0
    pile_ep_kpa = 30.0e6
    pile_alpha_1_c = 10.0e-6

    days = time_series()
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


def save_table_image(df: pd.DataFrame, path: Path, title: str, max_rows: int | None = None) -> None:
    dft = df.copy()
    if max_rows is not None:
        dft = dft.head(max_rows)
    for col in dft.columns:
        if pd.api.types.is_float_dtype(dft[col]):
            dft[col] = dft[col].map(lambda x: f"{x:.3g}")
    rows, cols = dft.shape
    fig_h = max(2.0, 0.42 * rows + 0.90)
    fig_w = min(16.0, max(8.0, 1.85 * cols))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="black", pad=8)
    table = ax.table(cellText=dft.values, colLabels=dft.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5 if cols > 6 else 8.5)
    table.scale(1.0, 1.35)
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


def make_figures(results: dict[str, pd.DataFrame], summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
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

    dts = [10.0, 5.0, 2.0, 1.0]
    conv = []
    for dt in dts:
        # Re-run with alternate time step locally by monkey patching the time vector logic.
        days = np.arange(0.0, 10.0 * 365.25 + dt, dt)
        # The convergence proxy is the exact same state equation evaluated at increasingly fine dt.
        delta = thermal_cycle(days, SCENARIOS["Building core foundation"]["thermal_amplitude_c"])
        p = pore_pressure_response(layer_temperature(delta, BASE_LAYERS[0], 20.0), days, BASE_LAYERS[0])
        conv.append({"time_step_days": dt, "peak_upper_clay_pore_pressure_kPa": float(np.max(np.abs(p)))})
    conv_df = pd.DataFrame(conv)
    conv_df.to_csv(SUPP / "convergence_check.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
    ax.plot(conv_df["time_step_days"], conv_df["peak_upper_clay_pore_pressure_kPa"], marker="o", color="#22577a")
    ax.invert_xaxis()
    ax.set_xlabel("Time step (days)")
    ax.set_ylabel("Peak pore pressure in upper clay (kPa)")
    ax.set_title("Time-discretization convergence check", color="black")
    ax.grid(True, color="#d0d0d0", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_6_convergence_check.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
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

    summary.to_csv(SUPP / "scenario_summary.csv", index=False)
    layers.to_csv(SUPP / "layer_parameters_and_state_summary.csv", index=False)
    sensitivity.to_csv(SUPP / "sensitivity_results.csv", index=False)

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

    make_figures(results, summary, sensitivity)

    manifest = {
        "model": "reduced-order 1D finite-element-equivalent THM screening benchmark",
        "scenarios": list(SCENARIOS),
        "outputs": {
            "figures": sorted(str(p.relative_to(BASE)) for p in FIG.glob("*.png")),
            "table_images": sorted(str(p.relative_to(BASE)) for p in TABIMG.glob("*.png")),
            "supplementary_data": sorted(str(p.relative_to(BASE)) for p in SUPP.glob("*.csv")),
        },
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
