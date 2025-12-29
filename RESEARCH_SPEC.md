# FMPT Research Specification

*Factorized Minimax Post-Training Under Reward, Domain, and Policy Shift*

## 1. Objective and contribution boundary

The project asks whether separately modeling three shifts gives a useful post-training guarantee:

1. the mixture of task domains changes;
2. the evaluator or reward channel changes within a domain;
3. the optimized policy produces responses outside the reference policy's support.

The intended contribution is not a new uncertainty estimator, reward ensemble, or generic application of GroupDRO. It is a rectangular ambiguity model tied to the post-training data-generating process, an exact dual when the shift sets are KL balls, domain-robust policy-divergence control, and matched DPO/GRPO specializations evaluated under the same controlled shifts.

## 2. Setup

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| $g\in\mathcal G$ | Task domain or prespecified reasoning group |
| $e\in\mathcal E_g$ | Evaluator, preference source, parser, verifier, or judge available in domain $g$ |
| $x\sim P_g$ | Prompt sampled from domain $g$ |
| $y\sim\pi_\theta(\cdot\mid x)$ | Response sampled from the optimized policy |
| $\pi_0=\pi_{\mathrm{SFT}}$ | Frozen SFT reference policy |
| $\widehat p_g$ | Empirical reference mass of domain $g$ |
| $\widehat a_{e\mid g}$ | Empirical reference mass or trust prior of evaluator $e$ in domain $g$ |
| $\widehat\mu_{g,e}=\widehat p_g\widehat a_{e\mid g}$ | Reference joint mass of a domain-evaluator cell |
| $L_{g,e}(\theta)$ | Domain-evaluator population loss |
| $K_g(\theta)$ | Expected policy KL from $\pi_0$ on prompts in domain $g$ |
| $\rho_d,\rho_r$ | Domain- and evaluator-shift radii |
| $\lambda_{\mathrm{KL}}$ | Policy-drift penalty relative to $\pi_0$ |

### 2.2 Reference factorization

The reference process is represented as

$$
\widehat P(g,e,x,y)
=\widehat p_g\,\widehat a_{e\mid g}\,\widehat P_g(x)\,\pi_0(y\mid x).
$$

The deployment process may change the domain mixture, the conditional evaluator mixture, and the response policy. Prompt covariate shift within a declared domain is outside the first theorem and is tested empirically as an extension.

### 2.3 Rectangular ambiguity set

For finite $\mathcal G$ and $\mathcal E_g$, define

$$
\mathcal U_d(\widehat p,\rho_d)
=\lbrace q\in\Delta^{\lvert\mathcal{G}\rvert}:D_{\mathrm{KL}}(q\Vert\widehat{p})\leq\rho_d\rbrace,
$$

$$
\mathcal U_{r,g}(\widehat a_g,\rho_r)
=\lbrace \nu_g\in\Delta^{\lvert\mathcal{E}_g\rvert}:D_{\mathrm{KL}}(\nu_g\Vert\widehat{a}_g)\leq\rho_r\rbrace,
$$

and the rectangular product set

$$
\mathfrak U(\rho_d,\rho_r)
=\lbrace (q,\nu):
D_{\mathrm{KL}}(q\Vert\widehat{p})\leq\rho_d,
\quad
D_{\mathrm{KL}}(\nu_g\Vert\widehat{a}_g)\leq\rho_r
\;\forall g
\rbrace.
$$

The per-domain constraint prevents an adversary from spending all evaluator uncertainty in a single high-mass domain while declaring every other evaluator known. A secondary ablation replaces the per-domain condition with an average conditional-divergence budget.

## 3. FMPT objective

### Definition 1: robust cell risk

$$
\mathcal R_{\rho_d,\rho_r}(\theta)
=\sup_{(q,\nu)\in\mathfrak U(\rho_d,\rho_r)}
\sum_g q_g\sum_e\nu_{g,e}L_{g,e}(\theta).
$$

### Definition 2: domainwise policy displacement

$$
K_g(\theta)
=\mathbb E_{x\sim\widehat P_g}
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\Vert\pi_0(\cdot\mid x)\right).
$$

### Definition 3: post-training objective

$$
\mathcal J_{\mathrm{FMPT}}(\theta)
=\sup_{q\in\mathcal U_d}
\sum_g q_g
\left[
\sup_{\nu_g\in\mathcal U_{r,g}}
\sum_e\nu_{g,e}L_{g,e}(\theta)
+\lambda_{\mathrm{KL}}K_g(\theta)
\right].
$$

The policy term is inside the domain adversary but outside the evaluator adversary. Thus a shifted domain mixture cannot conceal large drift on a rare group, and evaluator multiplicity does not multiply the KL penalty.

### Candidate dual

For fixed $\theta$, finite support, and a strictly feasible ambiguity set, the inner evaluator risk should admit

$$
\phi_g(\theta)
=\inf_{\eta_{r,g}>0}
\left[
\eta_{r,g}\rho_r
+\eta_{r,g}\log\sum_e\widehat{a}_{e\mid g}
\exp\!\left(\frac{L_{g,e}(\theta)}{\eta_{r,g}}\right)
\right]
$$

Define $\psi_g(\theta)=\phi_g(\theta)+\lambda_{\mathrm{KL}}K_g(\theta)$. The complete objective should then satisfy

$$
\mathcal J_{\mathrm{FMPT}}(\theta)
=\inf_{\eta_d>0}
\left[
\eta_d\rho_d
+\eta_d\log\sum_g\widehat p_g
\exp\!\left(\frac{\psi_g(\theta)}{\eta_d}\right)
\right].
$$

The nested entropic dual above follows from convex duality on finite support; boundary cases with zero radius or inactive constraints are handled explicitly.

## 4. Training specializations

### 4.1 FM-DPO

For an evaluator-conditioned preference tuple $(x,y^+,y^-)\sim P_{g,e}$, define

$$
m_\theta(x,y^+,y^-)
=\tau\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_0(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_0(y^-\mid x)}
\right]
$$

and

$$
L^{\mathrm{DPO}}_{g,e}(\theta)
=\mathbb E_{P_{g,e}}\left[-\log\sigma\!\left(m_\theta(x,y^+,y^-)\right)\right].
$$

FM-DPO substitutes these cell losses into Definition 3. Evaluator identity changes the preference distribution; it is not appended as a token to the prompt. The default implementation uses the same policy initialization, optimizer budget, pair set, and reference log-probabilities for every DPO baseline.

### 4.2 FM-GRPO

For prompt $x_b$, sample $K$ completions $y_{b,1:K}$. Evaluator $e$ returns rewards $r_{b,1:K,e}$, from which its own group-relative advantages are computed:

$$
A_{b,i,e}
=\frac{r_{b,i,e}-K^{-1}\sum_j r_{b,j,e}}
{\sqrt{K^{-1}\sum_j(r_{b,j,e}-\overline r_{b,e})^2}+\epsilon}.
$$

Let $L^{\mathrm{GRPO}}_{g,e}$ be the negative clipped policy-ratio surrogate using $A_{b,i,e}$ on prompts in domain $g$. FM-GRPO applies evaluator weights to these evaluator-specific surrogates and domain weights to the resulting domain losses. The reference-policy KL sits outside the evaluator adversary but inside the domain adversary, so it is counted once per domain.

If every completion in an evaluator group receives the same reward, that group contributes zero relative advantage and is logged as a reward-variance collapse. FMPT does not manufacture a learning signal in this case.

## 5. Proof program

Each item must have a formal statement, all assumptions, a complete appendix proof, and an executable finite-case check where possible.

### Proof target A: componentwise coverage

Let an audited failure loss $\ell_{g,e}(x,y)\in[0,M]$ be measured under $\pi_0$, and let $L^0_{g,e}$ denote its reference-policy expectation. For any admissible target mixtures $(q,\nu)$, Pinsker's inequality suggests the certificate target

$$
\mathbb E_{q,\nu,P_g,\pi_\theta}[\ell]
\leq
\sup_{q\in\mathcal U_d}
\sum_g q_g
\left[
\sup_{\nu_g\in\mathcal U_{r,g}}
\sum_e\nu_{g,e}L^0_{g,e}
+M\sqrt{K_g(\theta)/2}
\right].
$$

For every $\lambda>0$, Young's inequality gives

$$
M\sqrt{K_g(\theta)/2}
\leq \lambda K_g(\theta)+\frac{M^2}{8\lambda},
$$

which connects the linear domain-robust KL penalty to an additive certificate. The proof must state support assumptions and the direction of KL. Because DPO and GRPO optimize surrogates rather than the audited bounded loss, no accuracy guarantee follows unless a separate domination or calibration lemma is proved; otherwise this result is reported only as a reward-reliability certificate.

### Proof target B: exact duality

Prove the nested entropic dual above using convex duality. Characterize inactive constraints and show existence of an optimizer on finite support. Verify that moving $K_g$ outside the domain adversary generally breaks the intended worst-domain policy control.

### Proof target C: structured tightness

For $\mu_{q,\nu}(g,e)=q_g\nu_{g,e}$, use the KL chain rule to show

$$
 D_{\mathrm{KL}}(\mu_{q,\nu}\Vert\widehat{\mu})
 =D_{\mathrm{KL}}(q\Vert\widehat{p})
 +\sum_g q_g D_{\mathrm{KL}}(\nu_g\Vert\widehat{a}_g)
 \leq\rho_d+\rho_r.
$$

Then show that the cell-mixture distributions induced by $\mathfrak U(\rho_d,\rho_r)$ form a strict subset of the joint KL ball for at least one finite, full-support reference distribution and nondegenerate radii. Compare the resulting worst-case risks to explain when the joint ball is unnecessarily conservative.

### Proof target D: recovery and monotonicity

- $\rho_d=\rho_r=\lambda_{\mathrm{KL}}=0$: empirical mixture risk.
- $\rho_d=\rho_r=0$: empirical-mixture post-training with domain-mean policy KL.
- one domain: reward/evaluator pessimism plus policy control.
- one evaluator per domain: domain/group robustness plus domain-robust policy control.
- increasing either radius cannot lower robust loss.
- the unconstrained-radius limit approaches the worst supported cell.

### Proof target E: optimization

For bounded convex cell losses (or losses restricted to a compact parameter domain) and a log-linear policy, analyze alternating policy descent with exponentiated-gradient updates for $q$ and $\nu$. Target an $O(T^{-1/2})$ averaged saddle-gap statement. Neural-network training receives empirical stability analysis only.

## 6. Minimal counterexample

Construct a $2\times2$ loss matrix with two domains and two evaluators. The example must satisfy all of the following:

1. empirical-risk minimization selects a policy with the best mean loss;
2. domain-only robustness fails after evaluator shift;
3. evaluator-only pessimism fails after domain reweighting;
4. the factorized adversary selects a different policy with lower worst-case loss;
5. a single joint ball becomes more conservative than the rectangular set for the same stated componentwise budgets.

The example will be solved by exhaustive enumeration and included beside the analytic derivation.

## 7. Empirical protocol

### 8.1 Models

- `Qwen/Qwen3-0.6B-Base`: pilot, debugging, and full ablation model.
- `Qwen/Qwen3-1.7B-Base`: scale replication model.
- One SFT checkpoint is frozen for each size and used as the initialization and reference for every downstream method.
- Full-parameter BF16 training is the target comparison. Parameter-efficient pilots may debug the pipeline but cannot replace the matched final comparison.
- Neural experiments use seeds 17, 29, and 41. Seed deletion is forbidden; a failed run is reported and rerun only after a documented implementation fix applied to every method.

### 8.2 Reasoning data

The initial training pool contains a stratified 8,000-example mixture drawn from the training partitions of GSM8K and MATH after normalization and decontamination. The exact manifest must record source revision, license, example ID, original split, domain, subject, difficulty, normalization transform, and content hash.

Evaluation uses immutable, license-cleared revisions of:

- GSM8K test;
- MATH-500;
- GSM-Hard;
- AIME 2024 and 2025;
- AMC 2023;
- held-out MATH subjects and difficulty strata.

Before training, normalized n-gram and exact-answer overlap checks run between the training pool and every evaluation set. Suspected overlaps are quarantined, not silently deleted.

### 8.3 Candidate generation and preference pairs

For each SFT checkpoint and training prompt, generate eight completions from a frozen sampling configuration. Store token IDs and log-probabilities. Construct DPO pairs only when the clean verifier distinguishes a correct and incorrect completion; cap pairs per prompt to prevent easy prompts from dominating. Split by prompt before pair construction.

The GRPO lane uses the same prompt manifest and an eight-completion rollout group. DPO and GRPO compute budgets are reported separately; matching examples does not imply matching optimization steps.

### 8.4 Evaluator channels

The deterministic initial family is:

1. normalized exact-string/boxed-answer matching;
2. numeric equivalence with canonical fraction and tolerance handling;
3. symbolic equivalence with sandboxed parsing and a fixed timeout.

Each channel returns `score`, `valid`, `error_code`, and `evaluator_version`. Invalid or timed-out evaluations are missing observations, not automatic zeros, until a named scenario explicitly defines that behavior.

The held-out gold decision is produced by a stricter consensus rule plus manual audit of all evaluator disagreements in the primary test slice. A training evaluator may not serve as the sole gold evaluator.

### 8.5 Controlled shifts

1. **Symmetric reward noise:** independently flip binary outcomes at 10%, 20%, and 30% using stored corruption masks.
2. **Asymmetric evaluator noise:** set domain-conditional false-positive/false-negative rates, including noncanonical-but-correct answers and parser-friendly incorrect answers.
3. **Evaluator replacement:** swap the parser/equivalence implementation after training without changing model outputs.
4. **Domain reweighting:** evaluate fixed policies under preregistered mixtures that emphasize each minority subject and difficulty group.
5. **Held-out domain:** evaluate on reasoning sets absent from SFT and preference/RL prompts.
6. **Policy shift:** evaluate reward error on SFT, early, middle, and final checkpoints and in bins of realized response-level KL.
7. **Reward hacking:** apply semantics-audited transformations involving duplicated answers, competing boxed spans, formatting tokens, irrelevant verbosity, and answer-copy cues.

The primary hard setting combines domain reweighting, asymmetric evaluator noise, and the final policy checkpoint. Single-axis settings diagnose mechanisms; they are not substitutes for the combined test.

## 9. Baselines and ablations

### 9.1 Offline lane

- SFT reference.
- vanilla DPO.
- a label-noise-robust DPO baseline.
- Group Robust Preference Optimization, referred to as **Group-Robust DPO** to avoid collision with Group Relative Policy Optimization.
- KL-distributionally robust DPO and distributionally robust RLHF/DPO where implementation and data assumptions match.
- FM-DPO.

### 9.2 Online lane

- SFT reference.
- vanilla Group Relative Policy Optimization.
- prompt-GroupDRO and rollout-allocation GroupDRO for reasoning.
- mean reward aggregation.
- pessimistic/distributional reward aggregation.
- distributionally robust regret optimization where its reward-law assumptions match.
- false-positive/false-negative reward correction.
- FM-GRPO.

### 9.3 FMPT ablations

- `rho_domain = 0`;
- `rho_evaluator = 0`;
- `lambda_policy_kl = 0`;
- one joint KL ball with radius selected on the same validation budget;
- uniform rather than reliability-calibrated evaluator prior;
- fixed versus adaptive adversarial weights;
- per-domain versus averaged conditional evaluator budgets.

## 10. Metrics and estimands

### Primary estimands

1. average clean exact-match accuracy;
2. worst-domain accuracy;
3. worst domain-evaluator-cell accuracy;
4. shifted gold accuracy in the combined hard setting;
5. proxy-minus-gold reward gap as a function of realized policy KL.

### Safety and reliability

- invalid-format and policy-violation rates;
- reward-hacking success rate under each audited transformation;
- Brier score, expected calibration error, NLL where probabilistic, and AUROC for reward-error detection;
- held-out SFT token loss and clean benchmark regression;
- mean/max token KL, policy entropy, response length, reward variance, clip fraction, gradient norm, NaN/overflow count, and wall-clock/GPU-hours.

### Statistical reporting

- Report all three seed values and mean ± standard deviation.
- Use paired prompt-level bootstrap intervals for accuracy and hacking-rate differences.
- Use a stratified bootstrap for worst-group metrics so group identity is preserved.
- Treat model size as a replication axis, not as extra independent seeds.
- Correct the family of primary FMPT-vs-strongest-baseline tests with Holm's procedure.
- Report exact hyperparameter search spaces and total search cost for every method.

## 11. Acceptance and stop/go rules

The main empirical hypothesis passes only if FMPT beats the strongest eligible robust baseline on the combined-shift worst-cell metric at both model sizes, the paired interval excludes zero, and clean average accuracy is non-inferior within one absolute percentage point. A reward-hacking reduction without gold-accuracy retention is not a pass.

Do not submit a state-of-the-art claim if any of the following holds:

- a closer contemporaneous method already instantiates the same factorized ambiguity set;
- the central duality or coverage proof is incomplete;
- the result depends on excluding failed seeds;
- the held-out gold evaluator shares the exploited training-parser failure;
- the gain disappears against the strongest matched baseline or under the second model size.

Negative results should still be released if the protocol is intact. In that case the paper becomes an analysis of why single- and multi-axis robustness fail rather than a method paper.

## 12. Reproducibility and artifact schema

Every run manifest must include:

```yaml
run_id: string
method: sft | dpo | fm_dpo | grpo | fm_grpo | baseline_name
model_id: string
model_revision: git_sha_or_hf_revision
data_manifest_sha256: string
evaluator_manifest_sha256: string
seed: integer
rho_domain: nonnegative_float
rho_evaluator: nonnegative_float
lambda_policy_kl: nonnegative_float
config_sha256: string
code_commit: git_sha
```

Per-example JSONL stores prompt ID, domain/group IDs, completion token IDs, decoded text, per-evaluator scores/errors, gold outcome, policy/reference log-probabilities, realized KL, checkpoint, seed, and scenario ID. Aggregates are regenerated from this record.
