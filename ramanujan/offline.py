"""Offline demo mode: a scripted research session served through MockLLM.

`ramanujan run tasks/demo_breast_cancer.yaml --offline` exercises the ENTIRE
system - orchestration, real sandboxed sklearn training, ledger, report -
without any API key. Only the LLM responses are canned; the training code they
contain genuinely executes, and iteration 2 includes a deliberate ImportError
so the engineer's debug loop runs for real.
"""

from __future__ import annotations

import json

from .llm.base import LLMResponse, ToolCall
from .llm.mock import MockLLM

DEMO_TASK_SLUG = "breast-cancer-diagnosis"

_CODE_BASELINE = '''\
import json

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
print(f"data: {X.shape[0]} samples, {X.shape[1]} features, positive rate {y.mean():.3f}")

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"fold AUCs: {np.round(scores, 4).tolist()}")

mean_auc = float(scores.mean())
print(f"mean roc_auc: {mean_auc:.4f}")
with open("metrics.json", "w") as f:
    json.dump(
        {
            "metric_name": "roc_auc",
            "metric_value": mean_auc,
            "fold_std": float(scores.std()),
            "n_folds": 5,
        },
        f,
    )
'''

# Deliberately broken: HistGradientBoosting does not exist (the class is
# HistGradientBoostingClassifier). Exercises the real debug loop.
_CODE_BOOSTING_BUGGY = '''\
import json

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoosting
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
model = HistGradientBoosting(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
mean_auc = float(scores.mean())
print(f"mean roc_auc: {mean_auc:.4f}")
with open("metrics.json", "w") as f:
    json.dump({"metric_name": "roc_auc", "metric_value": mean_auc}, f)
'''

_CODE_BOOSTING_FIXED = '''\
import json

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
print(f"data: {X.shape[0]} samples, {X.shape[1]} features")

model = HistGradientBoostingClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"fold AUCs: {np.round(scores, 4).tolist()}")

mean_auc = float(scores.mean())
print(f"mean roc_auc: {mean_auc:.4f}")
with open("metrics.json", "w") as f:
    json.dump(
        {
            "metric_name": "roc_auc",
            "metric_value": mean_auc,
            "fold_std": float(scores.std()),
            "n_folds": 5,
        },
        f,
    )
'''


_CODE_EDA = '''\
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

print("=== SHAPE ===")
print(f"{X.shape[0]} samples, {X.shape[1]} numeric features")

print("=== TARGET BALANCE ===")
print(f"benign (1): {int(y.sum())}, malignant (0): {int((1 - y).sum())}, positive rate {y.mean():.3f}")

print("=== MISSING VALUES ===")
print(f"NaN count: {int(np.isnan(X).sum())}")

print("=== FEATURE SCALES ===")
ranges = X.max(axis=0) - X.min(axis=0)
print(f"feature ranges span {ranges.min():.4f} to {ranges.max():.1f} "
      "-> scales differ by orders of magnitude; standardization required for scale-sensitive models")

print("=== TOP |CORRELATION| WITH TARGET ===")
correlations = sorted(
    ((abs(float(np.corrcoef(X[:, i], y)[0, 1])), name, float(np.corrcoef(X[:, i], y)[0, 1]))
     for i, name in enumerate(data.feature_names)),
    reverse=True,
)
for _, name, c in correlations[:8]:
    print(f"  {name}: {c:+.3f}")

print("=== REDUNDANCY CHECK ===")
worst_idx = [i for i, n in enumerate(data.feature_names) if n.startswith("worst")]
mean_idx = [i for i, n in enumerate(data.feature_names) if n.startswith("mean")]
pair_corr = np.corrcoef(X[:, mean_idx[0]], X[:, worst_idx[0]])[0, 1]
print(f"mean/worst variants of the same measurement are highly correlated "
      f"(e.g. mean radius vs worst radius: {pair_corr:.3f})")
'''


def _json(payload: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload))


def _tool(name: str, **arguments) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, arguments=arguments)])


def build_offline_llm() -> MockLLM:
    """Canned responses in the exact order the orchestrator consumes them."""
    responses: list[LLMResponse] = [
        # ---- EDA phase (write -> run -> summarize -> distill)
        _tool("write_file", filename="eda.py", content=_CODE_EDA),
        _tool("run_script"),
        LLMResponse(
            text="Explored the dataset: 569 samples, 30 numeric features, no missing "
            "values, moderate class imbalance (63% benign). Feature scales differ by "
            "orders of magnitude, so standardization is required. Several features "
            "correlate strongly with the target (|r| up to ~0.78) and the mean/worst "
            "variants are highly redundant. No single feature is predictive enough "
            "to suggest leakage."
        ),
        _json(
            {
                "summary": "569 samples x 30 numeric features, no missing values, 63/37 "
                "class split. Signal is strong and spread across correlated feature "
                "groups; feature scales vary by orders of magnitude.",
                "key_findings": [
                    "No missing values and no constant columns - no imputation needed.",
                    "Feature scales span orders of magnitude; scale-sensitive models "
                    "require standardization.",
                    "Multiple features correlate strongly with the target (|r| up to ~0.78).",
                    "mean/worst variants of the same measurements are highly redundant.",
                ],
                "data_quality_issues": [],
                "leakage_risks": [
                    "None observed: no single feature predicts the target near-perfectly."
                ],
                "modeling_recommendations": [
                    "Start with a standardized linear model - strong correlated signal "
                    "suggests near-linear separability.",
                    "Use stratified CV because of the 63/37 class imbalance.",
                ],
            }
        ),
        # ---- iteration 1: planner
        _json(
            {
                "hypothesis": "A regularized linear model on standardized features already "
                "captures most of the signal in this low-dimensional tabular dataset.",
                "approach": "Logistic regression (max_iter=5000) in a pipeline with "
                "StandardScaler, evaluated with stratified 5-fold cross-validated ROC-AUC, "
                "random_state=42 throughout.",
                "rationale": "Establishes a strong, honest baseline before spending budget "
                "on higher-capacity models.",
            }
        ),
        # ---- iteration 1: engineer (write -> run -> summarize)
        _tool("write_file", filename="train.py", content=_CODE_BASELINE),
        _tool("run_script"),
        LLMResponse(
            text="Implemented a StandardScaler + LogisticRegression pipeline evaluated with "
            "stratified 5-fold CV ROC-AUC (seeded). The baseline is strong; metrics were "
            "written to metrics.json."
        ),
        # ---- iteration 1: analyst
        _json(
            {
                "insight": "The linear baseline is already near-ceiling: standardized features "
                "make the classes almost linearly separable, so remaining headroom is small.",
                "hypothesis_supported": True,
                "suspicion": "",
                "next_directions": [
                    "Test whether a gradient-boosted tree model captures any non-linear residual signal.",
                    "If boosting does not beat the linear model, conclude the dataset is "
                    "effectively linear and stop.",
                ],
            }
        ),
        # ---- iteration 1: critic
        _json(
            {
                "decision": "continue",
                "reasoning": "A single strong baseline is not yet convincing evidence of the "
                "ceiling. One contrasting model family will confirm whether non-linear "
                "structure exists; budget comfortably allows it.",
                "concerns": [],
            }
        ),
        # ---- iteration 2: planner
        _json(
            {
                "hypothesis": "Gradient-boosted trees capture non-linear feature interactions "
                "that the linear baseline misses, improving cross-validated ROC-AUC.",
                "approach": "HistGradientBoostingClassifier with default capacity "
                "(random_state=42), same stratified 5-fold CV ROC-AUC protocol as the "
                "baseline for a fair comparison. No scaling needed for trees.",
                "rationale": "Boosted trees are the strongest standard family on small tabular "
                "data; comparing against the identical CV protocol isolates the model effect.",
            }
        ),
        # ---- iteration 2: engineer (buggy write -> failing run -> fix -> run -> summarize)
        _tool("write_file", filename="train.py", content=_CODE_BOOSTING_BUGGY),
        _tool("run_script"),
        _tool("write_file", filename="train.py", content=_CODE_BOOSTING_FIXED),
        _tool("run_script"),
        LLMResponse(
            text="First attempt failed with an ImportError - the class is "
            "HistGradientBoostingClassifier, not HistGradientBoosting. Fixed the import, "
            "re-ran the same stratified 5-fold CV protocol; metrics written to metrics.json."
        ),
        # ---- iteration 2: analyst
        _json(
            {
                "insight": "Boosted trees do not beat the linear baseline: the dataset's "
                "signal is essentially linear after standardization, so added model capacity "
                "buys nothing.",
                "hypothesis_supported": False,
                "suspicion": "",
                "next_directions": [
                    "Model-family search is exhausted for this budget; only marginal gains "
                    "from calibration or feature selection remain plausible.",
                ],
            }
        ),
        # ---- iteration 2: critic
        _json(
            {
                "decision": "stop_goal_met",
                "reasoning": "The linear baseline meets the target ROC-AUC with a sound "
                "cross-validation protocol, and a contrasting model family failed to beat "
                "it - the result is both good enough and credible. Further budget would be "
                "spent on noise-level differences.",
                "concerns": [],
            }
        ),
        # ---- final report: conclusions
        LLMResponse(
            text="Two experiments were sufficient to solve this task and to understand why. "
            "A standardized logistic-regression baseline reached the target ROC-AUC "
            "immediately, indicating that the Wisconsin breast-cancer features are almost "
            "linearly separable once put on a common scale. A gradient-boosted tree model - "
            "the strongest conventional alternative on small tabular data - failed to "
            "improve on the linear model under an identical stratified 5-fold protocol, "
            "which is strong evidence that little non-linear signal exists.\n\n"
            "The main limitation is dataset size: with 569 samples, fold-level variance is "
            "non-trivial, and conclusions about sub-0.1% AUC differences would be "
            "unwarranted. A human researcher extending this work should prioritize "
            "calibration quality and external validation over further model search, since "
            "ranking performance is effectively saturated."
        ),
    ]
    return MockLLM(responses=responses)
