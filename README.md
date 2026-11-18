# Smart EV Charging Scheduling and Power Optimization Using PSO

A modular Python implementation of **Particle Swarm Optimization (PSO)** for day-ahead **Smart Electric Vehicle (EV) Charging Scheduling** under Time-of-Use (ToU) electricity pricing, transformer capacity constraints, and grid load-leveling objectives.

## Overview

Uncoordinated ("plug-and-charge") EV charging concentrates power demand during evening peak hours, inflating electricity costs, increasing feeder load variance, and overloading distribution transformers. This project formulates EV charging scheduling as a constrained multi-objective optimization problem and solves it with a **feasible-projection PSO** engine that guarantees every EV reaches its target State of Charge (SoC) before departure.

## Repository Structure

```text
smart-ev-charging-pso/
├── config/
│   └── simulation_config.json    # Fleet size, transformer limit, & PSO hyperparameters
├── src/
│   ├── __init__.py
│   ├── ev_charging_model.py      # EV fleet generator, ToU tariff, & grid evaluation
│   ├── pso_optimizer.py          # PSO engine with capped-simplex energy repair
│   └── visualization.py          # Load profile, convergence, & SoC trajectory plots
├── tests/
│   └── test_pso.py               # Pytest unit tests for constraints & optimization
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── .gitignore
```

## Mathematical Formulation

For a fleet of $N$ EVs over a 24-hour scheduling horizon ($t \in \{0, \dots, 23\}$):

- **Decision Variables**: Charging power $P_{i,t} \in [0, P_i^{\max}]$ for EV $i$ at hour $t$.
- **Availability Constraint**: $P_{i,t} = 0$ for $t \notin [t_i^{\text{arr}}, t_i^{\text{dep}})$.
- **Energy & SoC Constraint**:
  $$\sum_{t=t_i^{\text{arr}}}^{t_i^{\text{dep}}-1} P_{i,t} \Delta t = \frac{(SoC_i^{\text{target}} - SoC_i^{\text{init}}) E_i^{\text{cap}}}{\eta_i}$$
- **Multi-Objective Fitness**:
  $$F(P) = w_1 \frac{J_{\text{cost}}}{J_{\text{cost}}^{\text{ref}}} + w_2 \frac{J_{\text{var}}}{J_{\text{var}}^{\text{ref}}} + w_3 \sum_{t=0}^{23} \max\left(0, \frac{L_t - L^{\max}}{L^{\max}}\right)^2$$
  where $J_{\text{cost}} = \sum_t c_t \sum_i P_{i,t}\Delta t$ is total charging cost, $J_{\text{var}} = \text{Var}(L_t)$ is total grid load variance, and $L_t = L_t^{\text{base}} + \sum_i P_{i,t}$.

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation**

   ```bash
   python main.py --config config/simulation_config.json --output-dir results
   ```

3. **Run unit tests**

   ```bash
   pytest -v
   ```

## Generated Outputs

Running `main.py` produces:
- `results/hourly_load_comparison.csv`: Hourly base load, ToU tariff, uncoordinated EV load, and PSO-optimized EV load.
- `results/load_profile_comparison.png`: 24-hour grid load comparison against the transformer limit and ToU price curve.
- `results/pso_convergence.png`: Fitness and peak-load convergence across PSO iterations.
- `results/ev_soc_trajectories.png`: State-of-Charge (SoC) progression for sampled EVs.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
