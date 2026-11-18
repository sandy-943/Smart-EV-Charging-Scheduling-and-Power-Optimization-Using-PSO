"""CLI entry point for Smart EV Charging Scheduling and Power Optimization Using PSO."""

import argparse
import json
from pathlib import Path
import pandas as pd

from src.ev_charging_model import (
    evaluate_schedule,
    generate_ev_fleet,
    generate_grid_profile,
    uncoordinated_charging_schedule,
)
from src.pso_optimizer import PSOConfig, PSOEVScheduler
from src.visualization import (
    plot_convergence_curve,
    plot_load_comparison,
    plot_soc_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart EV Charging Scheduling & Power Optimization Using PSO"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/simulation_config.json",
        help="Path to simulation JSON config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to store output plots and CSV summaries.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sim_cfg = cfg["simulation"]
    pso_cfg_dict = cfg["pso"]
    weights = cfg["objective_weights"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(sim_cfg["horizon_hours"])
    transformer_limit = float(sim_cfg["transformer_limit_kw"])
    seed = int(sim_cfg["random_seed"])

    fleet = generate_ev_fleet(
        num_evs=int(sim_cfg["num_evs"]),
        horizon_hours=horizon,
        seed=seed,
    )
    base_load_kw, tou_tariff, hour_labels = generate_grid_profile(
        horizon_hours=horizon,
        start_hour_label=int(sim_cfg["start_hour_label"]),
    )

    # 1. Baseline: Uncoordinated Charging
    uncoord_sched = uncoordinated_charging_schedule(fleet, horizon)
    uncoord_metrics = evaluate_schedule(
        uncoord_sched,
        fleet,
        base_load_kw,
        tou_tariff,
        transformer_limit,
        weights,
    )

    # 2. PSO Optimization
    pso_config = PSOConfig(
        num_particles=int(pso_cfg_dict["num_particles"]),
        max_iterations=int(pso_cfg_dict["max_iterations"]),
        w_max=float(pso_cfg_dict["w_max"]),
        w_min=float(pso_cfg_dict["w_min"]),
        c1=float(pso_cfg_dict["c1"]),
        c2=float(pso_cfg_dict["c2"]),
        v_max_ratio=float(pso_cfg_dict["v_max_ratio"]),
        seed=seed,
    )
    scheduler = PSOEVScheduler(
        fleet=fleet,
        base_load_kw=base_load_kw,
        tou_tariff=tou_tariff,
        transformer_limit_kw=transformer_limit,
        objective_weights=weights,
        pso_config=pso_config,
    )
    pso_sched, pso_metrics, history = scheduler.optimize()

    # 3. Print Quantitative Summary
    cost_saving_pct = (
        (uncoord_metrics["total_cost"] - pso_metrics["total_cost"])
        / uncoord_metrics["total_cost"]
        * 100.0
    )
    peak_reduction_pct = (
        (uncoord_metrics["peak_load_kw"] - pso_metrics["peak_load_kw"])
        / uncoord_metrics["peak_load_kw"]
        * 100.0
    )
    var_reduction_pct = (
        (uncoord_metrics["load_variance"] - pso_metrics["load_variance"])
        / uncoord_metrics["load_variance"]
        * 100.0
    )

    print("=== Smart EV Charging Optimization Summary ===")
    print(f"Fleet Size           : {len(fleet)} EVs | Transformer Limit: {transformer_limit:.1f} kW")
    print(f"Uncoordinated Cost   : ${uncoord_metrics['total_cost']:.2f} | Peak: {uncoord_metrics['peak_load_kw']:.2f} kW | PAR: {uncoord_metrics['par']:.3f} | Overload: {uncoord_metrics['overload_kwh']:.2f} kWh")
    print(f"PSO-Optimized Cost   : ${pso_metrics['total_cost']:.2f} | Peak: {pso_metrics['peak_load_kw']:.2f} kW | PAR: {pso_metrics['par']:.3f} | Overload: {pso_metrics['overload_kwh']:.2f} kWh")
    print(f"Improvements         : Cost -{cost_saving_pct:.2f}% | Peak -{peak_reduction_pct:.2f}% | Load Variance -{var_reduction_pct:.2f}%")

    # 4. Save Hourly Results CSV & Plots
    df_hourly = pd.DataFrame(
        {
            "hour": hour_labels,
            "tou_tariff_usd_kwh": tou_tariff,
            "base_load_kw": base_load_kw,
            "uncoordinated_ev_kw": uncoord_metrics["ev_load_kw"],
            "uncoordinated_total_kw": uncoord_metrics["total_load_kw"],
            "pso_ev_kw": pso_metrics["ev_load_kw"],
            "pso_total_kw": pso_metrics["total_load_kw"],
        }
    )
    df_hourly.to_csv(out_dir / "hourly_load_comparison.csv", index=False)

    plot_load_comparison(
        hour_labels=hour_labels,
        base_load_kw=base_load_kw,
        uncoord_load_kw=uncoord_metrics["total_load_kw"],
        pso_load_kw=pso_metrics["total_load_kw"],
        tou_tariff=tou_tariff,
        transformer_limit_kw=transformer_limit,
        output_path=out_dir / "load_profile_comparison.png",
    )
    plot_convergence_curve(history, out_dir / "pso_convergence.png")
    plot_soc_trajectories(pso_sched, fleet, hour_labels, out_dir / "ev_soc_trajectories.png")
    print(f"Saved CSV and plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
