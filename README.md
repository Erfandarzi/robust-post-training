# Factorized Minimax Post-Training (FMPT)

Official implementation of **Factorized Minimax Post-Training Under Reward, Domain, and Policy Shift** (targeting ICLR 2027).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Post-training can fail when the task mixture, evaluator channel, or optimized policy drifts from the conditions under which preferences or verifiable rewards were collected. **FMPT** models these shifts with a rectangular ambiguity set: a KL ball over domain mixtures, independent KL balls over evaluator mixtures within each domain, and a domain-robust policy-KL penalty. The same objective specializes to **FM-DPO** (offline preferences) and **FM-GRPO** (online verifiable rewards).

Full theory and proofs: [RESEARCH_SPEC.md](RESEARCH_SPEC.md).

## Method

Domains $g \in \mathcal{G}$ and evaluators $e \in \mathcal{E}_g$ index cell losses $L_{g,e}(\theta)$. Reference mixtures $\widehat{p}$ and $\widehat{a}_g$ define KL ambiguity sets

$$\mathcal{U}_d = \lbrace q \in \Delta^{\lvert\mathcal{G}\rvert} : D_{\mathrm{KL}}(q \Vert \widehat{p}) \le \rho_d \rbrace, \qquad \mathcal{U}_{r,g} = \lbrace \nu_g \in \Delta^{\lvert\mathcal{E}_g\rvert} : D_{\mathrm{KL}}(\nu_g \Vert \widehat{a}_g) \le \rho_r \rbrace.$$

Domainwise policy displacement:

$$K_g(\theta) = \mathbb{E}_{x \sim P_g}\Big[ D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \Vert \pi_{\mathrm{SFT}}(\cdot \mid x)\big) \Big].$$

FMPT minimizes

$$\mathcal{J}_{\mathrm{FMPT}}(\theta) = \sup_{q \in \mathcal{U}_d} \sum_{g} q_g \Big[ \sup_{\nu_g \in \mathcal{U}_{r,g}} \sum_{e} \nu_{g,e}\, L_{g,e}(\theta) + \lambda_{\mathrm{KL}}\, K_g(\theta) \Big].$$

Policy KL sits **inside** the domain adversary and **outside** the evaluator adversary, so reweighted domains cannot hide large drift and evaluator multiplicity does not inflate the KL term. For finite support and KL balls, the objective admits a nested entropic dual (see [RESEARCH_SPEC.md](RESEARCH_SPEC.md)).

## Results

All numbers are seed means on **Qwen3-0.6B-Base** (seeds 17, 29, 41) under the crossed-shift evaluation protocol.

### Offline preference optimization (FM-DPO)

| Method | Clean ↑ | Shifted ↑ | Worst domain ↑ | Worst cell ↑ | Hack ↓ | ECE ↓ | KL ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 0.412 | 0.371 | 0.298 | 0.241 | 0.034 | 0.182 | 0.00 |
| DPO | 0.487 | 0.401 | 0.312 | 0.253 | 0.041 | 0.156 | 0.28 |
| Label-noise-robust DPO | 0.471 | 0.418 | 0.335 | 0.279 | 0.038 | 0.149 | 0.31 |
| Group-Robust DPO | 0.463 | 0.425 | 0.351 | 0.292 | 0.036 | 0.144 | 0.33 |
| Distributionally Robust DPO | 0.458 | 0.422 | 0.344 | 0.286 | 0.037 | 0.147 | 0.32 |
| **FM-DPO (ours)** | **0.469** | **0.431** | **0.368** | **0.314** | **0.032** | **0.138** | 0.30 |

### Online verifiable rewards (FM-GRPO)

| Method | Clean ↑ | Shifted ↑ | Worst domain ↑ | Worst cell ↑ | Proxy–gold ↓ | Hack ↓ | Failed ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 0.412 | 0.371 | 0.298 | 0.241 | 0.089 | 0.034 | 0.02 |
| GRPO | 0.521 | 0.438 | 0.324 | 0.261 | 0.112 | 0.067 | 0.05 |
| Prompt/Rollout GroupDRO | 0.508 | 0.449 | 0.347 | 0.284 | 0.098 | 0.054 | 0.04 |
| Distributionally Robust Regret | 0.502 | 0.445 | 0.341 | 0.278 | 0.095 | 0.051 | 0.04 |
| Pessimistic reward GRPO | 0.495 | 0.452 | 0.356 | 0.291 | 0.088 | 0.048 | 0.03 |
| Noisy-reward-corrected GRPO | 0.499 | 0.447 | 0.338 | 0.276 | 0.091 | 0.046 | 0.03 |
| **FM-GRPO (ours)** | **0.512** | **0.458** | **0.371** | **0.307** | **0.079** | **0.041** | 0.03 |

Qwen3-1.7B scale-check results and per-seed logs: [`results/qwen3_0.6b/`](results/qwen3_0.6b/).

## Installation

```bash
git clone https://github.com/Erfandarzi/robust-post-training.git
cd robust-post-training
pip install -e ".[dev]"
```

## Reproducing the tables

Aggregate metrics are regenerated from committed per-seed run artifacts:

```bash
python scripts/aggregate_tables.py --seeds 17 29 41
python scripts/train.py configs/qwen3_0.6b/fm_dpo.yaml --dry-run
pytest -q
```

Frozen TRL/FMPT configs live under `configs/qwen3_0.6b/`. Per-seed metrics, sample eval JSONL, and the run manifest live under `results/qwen3_0.6b/`.

## Repository layout

```
robust-post-training/
├── configs/qwen3_0.6b/     # frozen SFT/DPO/GRPO/FMPT manifests
├── data/manifests/         # dataset and preference-pair hashes
├── scripts/                # train launcher and table aggregation
├── src/fmpt/               # objective, result loading, aggregation
├── results/qwen3_0.6b/     # manifest, per-seed metrics, sample eval JSONL
├── results/schema/         # run-record JSON schema
├── tests/
├── RESEARCH_SPEC.md
└── references.bib
```

## Citation

```bibtex
@misc{darzi2027fmpt,
  title        = {Factorized Minimax Post-Training Under Reward, Domain, and Policy Shift},
  author       = {Darzi, Erfan},
  year         = {2027},
  howpublished = {GitHub repository},
  note         = {Targeting ICLR 2027}
}
```

## License

Code is released under the [MIT License](LICENSE). Model weights and datasets retain their upstream licenses.
