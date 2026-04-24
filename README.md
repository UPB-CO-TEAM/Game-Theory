# Two-Person Zero-Sum Game Analysis via Linear Programming

> Academic project developed for the **Operations Research** course  
> Faculty of Applied Sciences

---

## Table of Contents

1. [Overview](#overview)
2. [Authors](#authors)
3. [Methodology](#methodology)
4. [Validation System](#validation-system)
5. [Technologies & Architecture](#technologies--architecture)
6. [Usage Guide](#usage-guide)
7. [License](#license)

---

## Overview

A system for solving **two-person zero-sum strategic games** using a hybrid approach that combines classical matrix analysis with Linear Programming optimization. The application determines optimal strategies (pure or mixed) and the game value, providing a didactic perspective on duality in operations research.

🔗 **[Open Application]()**

> **Note:** All in-code comments are written in **Romanian**.

> **Dependency:** This project integrates the Primal Simplex engine developed in a companion project.  
> 🔗 **[Primal Simplex Algorithm — Application](https://game-theory-project-ildd.streamlit.app/)**  
> The Simplex computation module (`simplex.py`) is directly reused here to solve the dual Linear Programming models.

---

## Authors

| Name | Group |
|------|-------|
| Dedu Anișoara-Nicoleta | 1333a |
| Dumitrescu Andreea Mihaela | 1333a |
| Iliescu Daria-Gabriela | 1333a |
| Lungu Ionela-Diana | 1333a |

---

## Methodology

The computation process is divided into three fundamental stages, covering the transition from elementary analysis to advanced mathematical modelling.

### A. Pure Strategy Analysis (Saddle Point)

The algorithm first checks for a stable pure-strategy solution by computing:

| Indicator | Formula | Meaning |
|-----------|---------|---------|
| Lower Value (Maximin) `v₁` | `maxᵢ(minⱼ qᵢⱼ)` | Minimum guaranteed gain for Player A |
| Upper Value (Minimax) `v₂` | `minⱼ(maxᵢ qᵢⱼ)` | Maximum acceptable loss for Player B |

**Equilibrium Condition:** If `v₁ = v₂`, the game has a Saddle Point and the optimal strategies are pure.

### B. Linear Programming Formulation (Mixed Strategies)

If no saddle point exists (`v₁ < v₂`), the problem is transformed into a dual optimization model:

**Matrix Shift**  
If the payoff matrix `Q` contains negative entries, a constant `k` is added to all elements to ensure compatibility with the Simplex Algorithm.

**Dual Model Construction**

| Model | Player | Objective |
|-------|--------|-----------|
| LPA | Player A | Minimization — guaranteeing a minimum gain |
| LPB | Player B | Maximization — limiting losses |

The LPB model is solved using the external `simplex.py` computation module.

### C. Game Solution Reconstruction

After obtaining the optimal solution from the Simplex tableaux:

- **Game Value `v`** — derived from the optimal objective function value, adjusted by the shift constant `k`.
- **Probability Vectors `X₀`, `Y₀`** — raw LP solutions transformed into relative frequencies representing strategy usage probabilities.

---

## Validation System

Three critical tests guarantee the accuracy of the computed solution:

### ✅ V1 — Normalization
Verifies that the probability vectors sum to one:

```
Σᵢ xᵢ = 1    and    Σⱼ yⱼ = 1
```

### ✅ V2 — Value Bounds Check
Verifies that the game value falls within the theoretical bounds:

```
v₁ ≤ v ≤ v₂
```

### ✅ V3 — Fundamental Matrix Relation
Validates the game value through direct matrix computation:

```
v = X₀ · Q · Y₀ᵀ
```

---

## Technologies & Architecture

| Component | Technology |
|-----------|------------|
| Core Engine | Python 3.x — external `simplex.py` module for LP solving |
| Mathematical Display | LaTeX rendering for LP model formulations |
| Frontend | Streamlit — reactive interface with real-time payoff matrix editing |

---

## Usage Guide

**Step 1 — Configuration**  
Define the dimensions of the payoff matrix (number of strategies for each player).

**Step 2 — Data Input**  
Enter the values in matrix `Q` using the integrated data editor.

**Step 3 — Execution**  
Launch the computation to automatically display the pure strategy analysis, intermediate Simplex tableaux, and final results.

**Step 4 — Interpretation**  
Results are presented as probabilities (fractions) and performance metrics.

---

## License

© 2024 — Dedu A., Dumitrescu A., Iliescu D., Lungu D.

Operations Research — Faculty of Applied Sciences.  
Developed for academic purposes. Use of the code for educational purposes is permitted with **proper attribution to the authors**.
