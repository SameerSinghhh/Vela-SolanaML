#!/usr/bin/env python3
import os
import json
import math
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dotenv support for API key loading
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional OpenAI client; script supports offline fallback
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# SHAP explicitly not used per user request


# =============================
# CONFIGURATION
# =============================

# Toggle transaction costs to match test_model_simplified.py behavior
ENABLE_TRANSACTION_COSTS = False  # Keep currently disabled by default

# LLM model and search settings
OPENAI_MODEL = os.getenv("REASONING_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Policy search configuration
NUM_INITIAL_POLICIES = 32
TOP_M = 8
NUM_REFINEMENT_ROUNDS = 5  # increase refinement depth
NUM_REFINEMENTS_PER_ROUND = 16
REFINEMENT_MAX_TOKENS = 1500

# Internal validation split ratio from the TRAIN slice
INTERNAL_TRAIN_RATIO = 0.8  # 80% internal train, 20% valid

# Balanced labeled sample size for grounding
BALANCED_SAMPLE_PER_CLASS = 40  # total ~80 rows

# Heuristic enumeration size (broad search)
NUM_ENUM_POLICIES = 3000

# Policy bank configuration
POLICY_BANK_PATH = 'policy_bank.json'
POLICY_BANK_SIZE = 20  # keep top N historically best by validation protocol

# Deterministic seeding for local generation
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)

# Prediction balance constraints (target ~50/50 UP vs DOWN)
PRED_UP_TARGET = 0.5
PRED_UP_SOFT_MIN = 0.46
PRED_UP_SOFT_MAX = 0.54
PRED_UP_HARD_MIN = 0.46
PRED_UP_HARD_MAX = 0.54


# =============================
# DATA LOADING AND PREP
# =============================

def load_and_prepare_data(feature_matrix_path: str) -> Tuple[pd.DataFrame, List[str]]:
    print("\n🚀 Reasoning Policy Backtest")
    print("=" * 60)

    print("📊 Loading feature matrix...")
    df = pd.read_csv(feature_matrix_path)
    print(f"Dataset shape: {df.shape}")

    # Convert date and sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Match feature selection from test_model_simplified.py
    exclude_cols = ['date', 'sol_close', 'sol_actual_next_day_return', 'btc_close', 'eth_close', 'target_next_day', 'target_next_day_rolling_mean_2d']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Handle missing values by forward/backward fill
    X_all = df[feature_cols].copy()
    if X_all.isnull().values.any():
        X_all = X_all.ffill().bfill()
        for c in feature_cols:
            df[c] = X_all[c]

    return df, feature_cols


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n🔄 Creating CHRONOLOGICAL train/test split (70% train, 30% test)...")
    split_idx = int(len(df) * train_ratio)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    print(f"Training period: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} days)")
    print(f"Testing period:  {test_data['date'].min().date()} to {test_data['date'].max().date()} ({len(test_data)} days)")
    return train_data, test_data


def internal_train_valid_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    internal_split_idx = int(len(train_df) * INTERNAL_TRAIN_RATIO)
    internal_train = train_df.iloc[:internal_split_idx].copy()
    valid = train_df.iloc[internal_split_idx:].copy()
    print(f"Internal TRAIN: {internal_train['date'].min().date()} → {internal_train['date'].max().date()} ({len(internal_train)} days)")
    print(f"VALID:          {valid['date'].min().date()} → {valid['date'].max().date()} ({len(valid)} days)")
    return internal_train, valid


def rolling_valid_slices(train_df: pd.DataFrame, k_folds: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create multiple rolling VALID slices within TRAIN for robust selection.
    Strategy: split TRAIN into k contiguous folds; for each i, INTERNAL=[:i*fold], VALID=[i*fold:(i+1)*fold].
    Ensures VALID always follows INTERNAL chronologically.
    """
    n = len(train_df)
    fold = max(1, n // k_folds)
    slices: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(1, k_folds+1):
        cut = min(n, i * fold)
        internal = train_df.iloc[:max(fold, cut - fold)].copy()
        valid = train_df.iloc[max(fold, cut - fold):cut].copy()
        if len(internal) > 0 and len(valid) > 0:
            slices.append((internal, valid))
    return slices


# =============================
# STATS, SAMPLES, AND QUANTILES
# =============================

def compute_feature_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for col in feature_cols:
        series = df[col].astype(float)
        quantiles = series.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to_dict()
        stats[col] = {
            'mean': float(series.mean()),
            'std': float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            'quantiles': {str(k): float(v) for k, v in quantiles.items()},
        }
    return stats


def sample_balanced_rows(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    up_rows = df[df['target_next_day'] == 1]
    down_rows = df[df['target_next_day'] == -1]
    n_up = min(per_class, len(up_rows))
    n_down = min(per_class, len(down_rows))
    up_sample = up_rows.sample(n=n_up, random_state=42) if n_up > 0 else up_rows
    down_sample = down_rows.sample(n=n_down, random_state=42) if n_down > 0 else down_rows
    sample_df = pd.concat([up_sample, down_sample], axis=0).sample(frac=1.0, random_state=42)
    return sample_df


# =============================
# POLICY FORMAT AND EVALUATION
# =============================

def map_quantile_thresholds(policy: Dict[str, Any], feature_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mapped = json.loads(json.dumps(policy))  # deep copy
    for rule in mapped.get('rules', []):
        for predicate in rule.get('if', []):
            if 'thr_q' in predicate and 'thr' not in predicate:
                q = predicate['thr_q']
                feat = predicate['feat']
                # Expect q as fraction (e.g., 0.8) or str key ('0.8')
                q_key = str(q if isinstance(q, (int, float)) else q)
                quantiles = feature_stats.get(feat, {}).get('quantiles', {})
                if q_key in quantiles:
                    predicate['thr'] = quantiles[q_key]
                else:
                    # Fallback to median if missing
                    predicate['thr'] = feature_stats.get(feat, {}).get('quantiles', {}).get('0.5', 0.0)
    return mapped


def evaluate_decision_list(policy: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    def pred_row(row) -> int:
        for rule in policy.get('rules', []):
            conditions = rule.get('if', [])
            satisfied = True
            for c in conditions:
                feat = c['feat']
                op = c['op']
                thr = c['thr']
                val = row[feat]
                if op == '<' and not (val < thr):
                    satisfied = False
                    break
                elif op == '<=' and not (val <= thr):
                    satisfied = False
                    break
                elif op == '>' and not (val > thr):
                    satisfied = False
                    break
                elif op == '>=' and not (val >= thr):
                    satisfied = False
                    break
            if satisfied:
                decision = rule.get('then', 'DOWN')
                return 1 if decision.upper() == 'UP' else -1
        default_decision = policy.get('default', 'DOWN')
        return 1 if default_decision.upper() == 'UP' else -1

    preds = X.apply(pred_row, axis=1).astype(int).values
    return preds


def compute_turnover(preds: np.ndarray) -> int:
    if len(preds) <= 1:
        return 0
    changes = np.sum(np.abs(np.diff(preds)) > 0)
    return int(changes)


# =============================
# TRADING SIMULATION (MIRROR)
# =============================

def run_simulation(
    dates: np.ndarray,
    predictions: np.ndarray,
    actual_returns_pct: np.ndarray,
    actual_targets: np.ndarray,
    output_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    transaction_cost_per_trade = 0.002 if ENABLE_TRANSACTION_COSTS else 0.0

    initial_capital = 10000.0
    ml_capital = initial_capital
    buy_hold_capital = initial_capital

    # Apply initial cost for buy & hold (half of round-trip)
    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)

    detailed_log: List[Dict[str, Any]] = []

    # Max drawdown tracking
    ml_peak = initial_capital
    buy_hold_peak = buy_hold_capital
    ml_max_drawdown = 0.0
    buy_hold_max_drawdown = 0.0

    for date, pred, actual_ret_pct, actual_target in zip(dates, predictions, actual_returns_pct, actual_targets):
        if pd.isna(actual_ret_pct) or pd.isna(actual_target):
            continue

        actual_ret = actual_ret_pct / 100.0
        strategy_return = actual_ret if pred == 1 else -actual_ret
        strategy_return_after_costs = strategy_return - transaction_cost_per_trade

        ml_before = ml_capital
        bh_before = buy_hold_capital

        ml_capital = ml_capital * (1 + strategy_return_after_costs)
        buy_hold_capital = buy_hold_capital * (1 + actual_ret)

        # Drawdowns
        if ml_capital > ml_peak:
            ml_peak = ml_capital
        else:
            ml_max_drawdown = max(ml_max_drawdown, (ml_peak - ml_capital) / ml_peak)

        if buy_hold_capital > buy_hold_peak:
            buy_hold_peak = buy_hold_capital
        else:
            buy_hold_max_drawdown = max(buy_hold_max_drawdown, (buy_hold_peak - buy_hold_capital) / buy_hold_peak)

        prediction_correct = int(pred == actual_target)

        detailed_log.append({
            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
            'actual_target': int(actual_target),
            'prediction': int(pred),
            'prediction_correct': bool(prediction_correct),
            'actual_return_pct': actual_ret * 100,
            'strategy_return_pct': strategy_return * 100,
            'strategy_return_after_costs_pct': strategy_return_after_costs * 100,
            'reasoning_balance_before': ml_before,
            'reasoning_balance_after': ml_capital,
            'buy_hold_balance_before': bh_before,
            'buy_hold_balance_after': buy_hold_capital,
        })

    # Final sell cost for buy & hold
    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)
    if detailed_log:
        detailed_log[-1]['buy_hold_balance_after'] = buy_hold_capital

    # Save CSV if requested
    if output_csv_path:
        pd.DataFrame(detailed_log).to_csv(output_csv_path, index=False)
        print(f"\n✅ Detailed daily results saved to: {output_csv_path}")

    # Daily returns for Sharpe
    ml_daily_returns: List[float] = []
    bh_daily_returns: List[float] = []
    for i in range(len(detailed_log)):
        if i == 0:
            ml_prev = detailed_log[i]['reasoning_balance_before']
            bh_prev = detailed_log[i]['buy_hold_balance_before']
        else:
            ml_prev = detailed_log[i-1]['reasoning_balance_after']
            bh_prev = detailed_log[i-1]['buy_hold_balance_after']
        ml_curr = detailed_log[i]['reasoning_balance_after']
        bh_curr = detailed_log[i]['buy_hold_balance_after']
        ml_daily_returns.append((ml_curr - ml_prev) / ml_prev)
        bh_daily_returns.append((bh_curr - bh_prev) / bh_prev)

    ml_mean = float(np.mean(ml_daily_returns)) if ml_daily_returns else 0.0
    ml_std = float(np.std(ml_daily_returns, ddof=1)) if len(ml_daily_returns) > 1 else 0.0
    bh_mean = float(np.mean(bh_daily_returns)) if bh_daily_returns else 0.0
    bh_std = float(np.std(bh_daily_returns, ddof=1)) if len(bh_daily_returns) > 1 else 0.0

    ml_sharpe_daily = (ml_mean / ml_std) if ml_std > 0 else 0.0
    bh_sharpe_daily = (bh_mean / bh_std) if bh_std > 0 else 0.0
    ml_sharpe_annual = ml_sharpe_daily * math.sqrt(252)
    bh_sharpe_annual = bh_sharpe_daily * math.sqrt(252)

    ml_total_return = (ml_capital - 10000.0) / 10000.0
    bh_total_return = (buy_hold_capital - 10000.0) / 10000.0

    total_days = len(detailed_log)
    winning_days = sum(1 for d in detailed_log if d['prediction_correct'])
    win_rate = (winning_days / total_days) if total_days > 0 else 0.0

    return {
        'ml_capital': ml_capital,
        'buy_hold_capital': buy_hold_capital,
        'ml_total_return': ml_total_return,
        'buy_hold_total_return': bh_total_return,
        'ml_sharpe_annual': ml_sharpe_annual,
        'buy_hold_sharpe_annual': bh_sharpe_annual,
        'ml_max_drawdown': ml_max_drawdown,
        'buy_hold_max_drawdown': buy_hold_max_drawdown,
        'detailed_log': detailed_log,
        'win_rate': win_rate,
        'winning_days': winning_days,
        'total_days': total_days,
    }


# =============================
# EVALUATION METRICS
# =============================

def classification_and_bias_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])

    # Class-specific accuracy
    results: Dict[str, Any] = {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist(),
        'counts': {
            'actual_up': int((y_true == 1).sum()),
            'actual_down': int((y_true == -1).sum()),
            'pred_up': int((y_pred == 1).sum()),
            'pred_down': int((y_pred == -1).sum()),
        }
    }

    for cls_val, cls_name in [(1, 'UP'), (-1, 'DOWN')]:
        mask = (y_true == cls_val)
        cls_acc = float((y_pred[mask] == cls_val).mean()) if mask.sum() > 0 else 0.0
        results[f'{cls_name.lower()}_accuracy'] = cls_acc

    return results


# =============================
# SHAP-INFORMED FEATURE RANKING
# =============================

def compute_shap_importance(internal_train: pd.DataFrame, feature_cols: List[str]) -> List[Tuple[str, float]]:
    if shap is None:
        print("⚠️  SHAP not available; skipping SHAP ranking.")
        return []

    try:
        X = internal_train[feature_cols].values
        y = internal_train['target_next_day'].values

        # Train simple SVM pipeline as proxy for SHAP
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ])
        pipeline.fit(X, y)

        # Background and sample
        np.random.seed(42)
        bg_idx = np.random.choice(len(X), size=min(100, len(X)), replace=False)
        sample_idx = np.random.choice(len(X), size=min(100, len(X)), replace=False)
        background = pipeline.named_steps['scaler'].transform(X[bg_idx])
        sample = pipeline.named_steps['scaler'].transform(X[sample_idx])

        explainer = shap.KernelExplainer(pipeline.named_steps['svc'].predict_proba, background)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_up = np.array(shap_values[1])
        else:
            shap_up = np.array(shap_values)

        mean_abs = np.mean(np.abs(shap_up), axis=0)
        mean_abs = np.asarray(mean_abs).flatten()
        if len(mean_abs) != len(feature_cols):
            L = min(len(mean_abs), len(feature_cols))
            ranked = sorted([(feature_cols[i], float(mean_abs[i])) for i in range(L)], key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(list(zip(feature_cols, mean_abs)), key=lambda x: x[1], reverse=True)

        with open('shap_importance.json', 'w') as f:
            json.dump([{ 'feature': k, 'importance': v } for k, v in ranked], f, indent=2)
        print("✅ SHAP importance saved to shap_importance.json")
        print("Top SHAP features:")
        for i, (feat, imp) in enumerate(ranked[:10], start=1):
            print(f"  {i:>2}. {feat:<35} {imp:.5f}")
        return ranked
    except Exception as e:
        print(f"⚠️  SHAP ranking failed: {e}")
        return []


# =============================
# LLM INTEGRATION
# =============================

def build_llm_context(feature_cols: List[str], stats: Dict[str, Dict[str, Any]], sample_df: pd.DataFrame) -> str:
    # Compact stats: per-feature mean, std, deciles
    lines: List[str] = []
    lines.append("You are proposing interpretable trading rules as a decision list.")
    lines.append("Only output valid JSON as an array of policy objects, no narration.")
    lines.append("Each policy must follow this schema strictly:")
    lines.append("{"
                 "\"policy_type\": \"decision_list\","
                 "\"rules\": [ { \"if\": [ {\"feat\": str, \"op\": one of ['<','<=','>','>='], \"thr\": number OR \"thr_q\": quantile float 0..1 } ], \"then\": 'UP'|'DOWN' } ],"
                 "\"default\": 'UP'|'DOWN',"
                 "\"constraints\": { \"max_rules\": 5, \"max_predicates_per_rule\": 3 }"
                 "}")
    lines.append("Allowed features:")
    lines.append(", ".join(feature_cols))
    lines.append("Feature stats (mean, std, q10..q90) computed on internal TRAIN:")
    for feat in feature_cols:
        s = stats[feat]
        qs = s['quantiles']
        q_str = ", ".join([f"q{int(float(k)*100)}:{qs[k]:.4f}" for k in sorted(qs.keys(), key=lambda x: float(x))])
        lines.append(f"{feat}: mean={s['mean']:.4f}, std={s['std']:.4f}, {q_str}")

    # Small labeled sample for grounding
    max_rows = min(len(sample_df), BALANCED_SAMPLE_PER_CLASS * 2)
    lines.append(f"Labeled sample (first {max_rows} rows; columns: features + target_next_day):")
    preview_cols = feature_cols + ['target_next_day']
    lines.extend(sample_df[preview_cols].head(max_rows).to_csv(index=False).splitlines())

    # Objective
    lines.append("Objective: propose diverse policies that maximize Sharpe (primary), with lower max drawdown as tie-breaker. Keep rules concise and robust (prefer quantile thresholds). Output exactly 32 policies.")
    return "\n".join(lines)


def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    if not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI()
        return client
    except Exception:
        return None


def request_policies_from_llm(client: Any, prompt: str, k: int) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You are an expert quant proposing concise, interpretable trading rules. Output JSON only."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Some models (e.g., 'o3-mini') do not support 'temperature'
        kwargs = {
            'model': OPENAI_MODEL,
            'messages': messages,
            'max_completion_tokens': 6000,
        }
        if not str(OPENAI_MODEL).lower().startswith('o3'):
            kwargs['temperature'] = 0.7
        res = client.chat.completions.create(**kwargs)
        content = res.choices[0].message.content
    except Exception as e:
        print(f"⚠️  LLM request failed: {e}")
        return []


def rolling_windows_by_days(df: pd.DataFrame, window_days: int = 21, step_days: int = 7, min_rows: int = 10) -> List[pd.DataFrame]:
    df = df.copy()
    df = df.sort_values('date')
    start_date = df['date'].min()
    end_date = df['date'].max()
    windows: List[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        w_end = current + pd.Timedelta(days=window_days)
        slice_df = df[(df['date'] >= current) & (df['date'] <= w_end)].copy()
        if len(slice_df) >= min_rows:
            windows.append(slice_df)
        current = current + pd.Timedelta(days=step_days)
    return windows


def request_rules_for_window(client: Any, window_df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Ask LLM to propose a handful of rules for a small time window to avoid overloading context."""
    # Build compact context
    stats = compute_feature_stats(window_df, feature_cols)
    sample = sample_balanced_rows(window_df, per_class=10)
    lines: List[str] = []
    lines.append("You will propose a small set of interpretable IF-THEN rules (no more than 5) for a decision-list trading policy.")
    lines.append("Only output valid JSON: an array of rule objects. No narration.")
    lines.append("Each rule: { 'if': [ { 'feat': str, 'op': '<'|'<=|'>'|'>=', 'thr': number }... ], 'then': 'UP'|'DOWN' }")
    lines.append("Use these features (subset shown for brevity; you may reference any provided in the sample):")
    lines.append(", ".join(feature_cols[:20]))
    lines.append("Window summary stats:")
    for f in feature_cols[:20]:
        s = stats[f]
        lines.append(f"{f}: mean={s['mean']:.3f} std={s['std']:.3f} q20={s['quantiles']['0.2']:.3f} q50={s['quantiles']['0.5']:.3f} q80={s['quantiles']['0.8']:.3f}")
    lines.append("Labeled sample (features + target):")
    lines.extend(sample[feature_cols + ['target_next_day']].head(20).to_csv(index=False).splitlines())
    lines.append("Constraints: prefer 1-3 predicates per rule; max 5 rules total; prefer quantile-like thresholds (around q20/q50/q80). Balance UP/DOWN; aim for ~50/50 predictions.")

    messages = [
        {"role": "system", "content": "Think step-by-step about patterns in this small window, then output JSON array of rules only."},
        {"role": "user", "content": "\n".join(lines)}
    ]

    try:
        kwargs = {
            'model': OPENAI_MODEL,
            'messages': messages,
            'max_completion_tokens': 1200,
        }
        if not str(OPENAI_MODEL).lower().startswith('o3'):
            kwargs['temperature'] = 0.3
        res = client.chat.completions.create(**kwargs)
        content = res.choices[0].message.content
        data = json.loads(content) if content.strip().startswith('[') else []
        return [r for r in data if isinstance(r, dict)]
    except Exception:
        return []


def collect_llm_rules_via_windows(client: Any, internal_train: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    if client is None:
        return rules
    wins = rolling_windows_by_days(internal_train, window_days=30, step_days=15, min_rows=10)
    for w in wins[:12]:  # cap calls for speed; can increase
        rules.extend(request_rules_for_window(client, w, feature_cols))
    return rules


def score_single_rule(rule: Dict[str, Any], df_slice: pd.DataFrame, feature_cols: List[str]) -> float:
    policy = { 'policy_type': 'decision_list', 'rules': [rule], 'default': 'DOWN' }
    res = evaluate_policy_objective(policy, df_slice, feature_cols)
    return res['sharpe']


def build_policies_from_rules(rules: List[Dict[str, Any]], internal_train: pd.DataFrame, feature_cols: List[str], top_rules: int = 30, combos: int = 256) -> List[Dict[str, Any]]:
    # De-duplicate rules by JSON signature
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in rules:
        try:
            key = json.dumps(r, sort_keys=True)
        except Exception:
            continue
        uniq[key] = r
    unique_rules = list(uniq.values())
    # Score rules individually
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in unique_rules:
        try:
            s = score_single_rule(r, internal_train, feature_cols)
            scored.append((s, r))
        except Exception:
            continue
    # Select top rules
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [r for _, r in scored[:top_rules]]
    if len(selected) < 2:
        return []
    # Build combo policies
    rng = np.random.default_rng(123)
    policies: List[Dict[str, Any]] = []
    for _ in range(combos):
        max_k = min(5, len(selected))
        if max_k < 2:
            break
        k = int(rng.integers(low=2, high=max_k+1))
        chosen = list(rng.choice(selected, size=k, replace=False))
        policy = { 'policy_type': 'decision_list', 'rules': chosen, 'default': 'DOWN', 'constraints': { 'max_rules': 8, 'max_predicates_per_rule': 6 } }
        policies.append(policy)
    return policies

    # Expect JSON array
    try:
        # Strict JSON: require an array
        data = json.loads(content)
        if isinstance(data, list):
            return data[:k]
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except Exception:
        # Attempt to extract JSON array by naive slicing once, then give up
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(content[start:end+1])
                if isinstance(data, list):
                    return data[:k]
            except Exception:
                pass
        print("⚠️  Failed to parse policies from LLM output; skipping LLM candidates this round.")
        return []


def validate_and_prepare_policies(raw_policies: List[Dict[str, Any]], feature_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for p in raw_policies:
        if not isinstance(p, dict):
            continue
        if p.get('policy_type') != 'decision_list':
            continue
        # Default constraints
        p.setdefault('constraints', {"max_rules": 5, "max_predicates_per_rule": 3})
        rules = p.get('rules', [])
        if not isinstance(rules, list) or len(rules) == 0:
            continue
        if len(rules) > 5:
            rules = rules[:5]
            p['rules'] = rules
        # Clip predicates per rule
        for r in rules:
            conditions = r.get('if', [])
            if not isinstance(conditions, list) or len(conditions) == 0:
                r['if'] = []
            if len(conditions) > 3:
                r['if'] = conditions[:3]
        # Map thr_q to thr using feature_stats
        mapped = map_quantile_thresholds(p, feature_stats)
        prepared.append(mapped)
    return prepared


# =============================
# SEARCH LOOP
# =============================

def evaluate_policy_objective(policy: Dict[str, Any], df_slice: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    X = df_slice[feature_cols]
    preds = evaluate_decision_list(policy, X)
    dates = df_slice['date'].values
    actual_returns_pct = df_slice['sol_actual_next_day_return'].values
    actual_targets = df_slice['target_next_day'].values

    sim = run_simulation(
        dates=dates,
        predictions=preds,
        actual_returns_pct=actual_returns_pct,
        actual_targets=actual_targets,
        output_csv_path=None,  # no printing inside
    )

    turnover = compute_turnover(preds)
    metrics = classification_and_bias_analysis(actual_targets, preds)
    # Prediction balance penalty (soft)
    total = len(preds)
    pred_up_share = float((preds == 1).sum()) / total if total > 0 else 0.5
    balance_penalty = 0.0
    if pred_up_share < PRED_UP_SOFT_MIN:
        balance_penalty = (PRED_UP_SOFT_MIN - pred_up_share)
    elif pred_up_share > PRED_UP_SOFT_MAX:
        balance_penalty = (pred_up_share - PRED_UP_SOFT_MAX)
    # Adjust Sharpe by penalty (small impact to steer away from extremes)
    adjusted_sharpe = metrics.get('accuracy', 0)  # placeholder not used further
    sharpe = sim['ml_sharpe_annual'] - 0.5 * balance_penalty
    return {
        'policy': policy,
        'sharpe': sharpe,
        'max_drawdown': sim['ml_max_drawdown'],
        'turnover': turnover,
        'pred_up_share': pred_up_share,
        'accuracy': metrics['accuracy'],
        'metrics': metrics,
    }


def select_top(results: List[Dict[str, Any]], top_m: int) -> List[Dict[str, Any]]:
    # Sort by Sharpe desc, then by max_drawdown asc, then by turnover asc
    sorted_res = sorted(results, key=lambda r: (-r['sharpe'], r['max_drawdown'], r['turnover']))
    return sorted_res[:top_m]


def evaluate_over_rolling_valid(policy: Dict[str, Any], train_df: pd.DataFrame, feature_cols: List[str], k_folds: int = 3) -> Dict[str, float]:
    """Evaluate a policy over multiple rolling VALID slices; return aggregated metrics used for selection."""
    slices = rolling_valid_slices(train_df, k_folds=k_folds)
    sharpes: List[float] = []
    maxdds: List[float] = []
    turnovers: List[int] = []
    for internal, valid in slices:
        X = valid[feature_cols]
        preds = evaluate_decision_list(policy, X)
        sim = run_simulation(
            dates=valid['date'].values,
            predictions=preds,
            actual_returns_pct=valid['sol_actual_next_day_return'].values,
            actual_targets=valid['target_next_day'].values,
            output_csv_path=None,
        )
        turnovers.append(compute_turnover(preds))
        sharpes.append(sim['ml_sharpe_annual'])
        maxdds.append(sim['ml_max_drawdown'])
    if not sharpes:
        return {'avg_sharpe': -1e9, 'avg_maxdd': 1e9, 'avg_turnover': 1e9}
    return {
        'avg_sharpe': float(np.mean(sharpes)),
        'avg_maxdd': float(np.mean(maxdds)),
        'avg_turnover': float(np.mean(turnovers)),
    }


def generate_policies_enumerated(
    feature_stats: Dict[str, Dict[str, Any]],
    feature_cols: List[str],
    num_policies: int,
    max_rules_cap: int = 5,
    max_predicates_cap: int = 3,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng()
    # Prefer a curated subset of commonly useful features if present
    preferred = [
        'sol_rsi_14', 'sol_macd_histogram', 'sol_return_1d', 'sol_return_3d', 'sol_return_7d',
        'sol_close_sma7_ratio', 'sol_sma7_sma14_ratio', 'sol_price_dev_from_sma7',
        'btc_return_1d', 'btc_return_3d', 'btc_return_7d',
        'eth_return_1d', 'eth_return_3d', 'eth_return_7d',
        'vix', 'dxy', 'fedfunds', 'sol_volatility_7d',
        'sol_price_relative_to_btc', 'sol_price_relative_to_eth'
    ]
    top_feats = [f for f in preferred if f in feature_cols]
    if len(top_feats) < 8:
        # augment with any remaining features
        remaining = [f for f in feature_cols if f not in top_feats]
        top_feats.extend(remaining[: max(0, 12 - len(top_feats))])
    quantile_opts = [0.2, 0.3, 0.5, 0.7, 0.8]
    policies: List[Dict[str, Any]] = []

    def sample_predicate() -> Dict[str, Any]:
        feat = rng.choice(top_feats)
        q = float(rng.choice(quantile_opts))
        thr = feature_stats.get(feat, {}).get('quantiles', {}).get(str(q), None)
        if thr is None:
            thr = feature_stats.get(feat, {}).get('mean', 0.0)
        op = rng.choice(['<', '<=', '>', '>='])
        if rng.random() < 0.5:
            return { 'feat': feat, 'op': op, 'thr_q': q }
        else:
            return { 'feat': feat, 'op': op, 'thr': float(thr) }

    for _ in range(num_policies):
        num_rules = int(rng.integers(low=2, high=max(3, max_rules_cap)+1))  # 2..cap rules
        rules: List[Dict[str, Any]] = []
        for _r in range(num_rules):
            num_preds = int(rng.integers(low=1, high=max(2, max_predicates_cap)+1))  # 1..cap predicates
            preds = []
            used_feats = set()
            for _p in range(num_preds):
                pr = sample_predicate()
                if pr['feat'] in used_feats:
                    continue
                used_feats.add(pr['feat'])
                preds.append(pr)
            # bias rules based on typical patterns
            if any(p.get('feat') == 'sol_rsi_14' and p.get('op') in ['<','<='] for p in preds):
                then = 'UP'
            elif any(p.get('feat') == 'sol_rsi_14' and p.get('op') in ['>','>='] for p in preds):
                then = 'DOWN'
            else:
                then = 'UP' if rng.random() < 0.5 else 'DOWN'
            rules.append({ 'if': preds, 'then': then })
        default = 'UP' if rng.random() < 0.5 else 'DOWN'
        policies.append({
            'policy_type': 'decision_list',
            'rules': rules,
            'default': default,
            'constraints': { 'max_rules': max_rules_cap, 'max_predicates_per_rule': max_predicates_cap }
        })
    # Enforce hard prediction balance on INTERNAL_TRAIN via quick screening
    screened: List[Dict[str, Any]] = []
    # Note: Screening will be applied later when evaluating; here we just return generated
    return policies


def reasoning_policy_search(
    internal_train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    print("\n🧠 Reasoning policy search (TRAIN only)")
    feature_stats = compute_feature_stats(internal_train, feature_cols)
    sample_df = sample_balanced_rows(internal_train, BALANCED_SAMPLE_PER_CLASS)
    context_prompt = build_llm_context(feature_cols, feature_stats, sample_df)

    client = get_openai_client()
    all_round_candidates: List[Dict[str, Any]] = []

    # Round 0: initial exploration (small-context windows + enumerated)
    if client is not None:
        # Collect granular rules over weekly windows, then compose policies
        print("Collecting windowed rules from LLM...")
        window_rules = collect_llm_rules_via_windows(client, internal_train, feature_cols)
        llm_policies = build_policies_from_rules(window_rules, internal_train, feature_cols, top_rules=20, combos=128)
        raw_policies = llm_policies
    else:
        print("⚠️  No OpenAI client available; using empty policy set. You can provide candidate policies via 'policies_round_0.json'.")
        raw_policies = []

    # Allow user-provided candidates as fallback
    if len(raw_policies) == 0 and os.path.exists('policies_round_0.json'):
        try:
            with open('policies_round_0.json', 'r') as f:
                raw_policies = json.load(f)
        except Exception:
            raw_policies = []

    prepared_policies = validate_and_prepare_policies(raw_policies, feature_stats)
    print(f"Round 0: received {len(prepared_policies)} policies from LLM/input")

    # Add enumerated policies (broad search, SHAP-free)
    # Try multiple caps and seeds to broaden search space
    enum_policies = []
    seed_base = 100
    for max_rules_cap in [5, 6, 7, 8]:
        for max_preds_cap in [3, 4, 5, 6]:
            for _s in range(3):
                enum_policies.extend(
                    generate_policies_enumerated(
                        feature_stats, feature_cols, NUM_ENUM_POLICIES // 6, max_rules_cap=max_rules_cap, max_predicates_cap=max_preds_cap
                    )
                )
    print(f"Round 0: generated {len(enum_policies)} enumerated policies")
    prepared_policies.extend(validate_and_prepare_policies(enum_policies, feature_stats))

    # Evaluate on INTERNAL_TRAIN (apply hard balance screen)
    round0_results: List[Dict[str, Any]] = []
    tested_count = 0
    filtered_balance_it = 0
    for p in prepared_policies:
        res = evaluate_policy_objective(p, internal_train, feature_cols)
        share = res.get('pred_up_share', 0.5)
        tested_count += 1
        if share < PRED_UP_HARD_MIN or share > PRED_UP_HARD_MAX:
            filtered_balance_it += 1
            continue
        round0_results.append(res)
    top_candidates = select_top(round0_results, TOP_M)
    print(f"Evaluated INTERNAL_TRAIN candidates: {tested_count}, filtered by balance: {filtered_balance_it}, retained: {len(round0_results)}")

    # Save round 0 logs
    with open('policies_round_0_results.json', 'w') as f:
        json.dump(top_candidates, f, indent=2)

    # Refinement rounds
    current_candidates = [tc['policy'] for tc in top_candidates]
    for r in range(NUM_REFINEMENT_ROUNDS):
        if client is None:
            print("⚠️  Skipping refinement; no OpenAI client available.")
            break

        # Build refinement prompt: show metrics, request N improved variants
        compact_summaries = []
        for c in top_candidates:
            pol = c['policy']
            compact_summaries.append({
                'sharpe': c['sharpe'],
                'max_drawdown': c['max_drawdown'],
                'turnover': c['turnover'],
                'accuracy': c['accuracy'],
                'policy': pol,
            })

        # Use a compact JSON payload to avoid token/JSON issues
        refine_payload = {
            'instruction': 'Propose N improved decision-list variants focusing on higher Sharpe and lower drawdown. Use quantile-based thresholds when possible. Keep JSON strict.',
            'N': NUM_REFINEMENTS_PER_ROUND,
            'constraints': {"max_rules": 7, "max_predicates_per_rule": 4},
            'top_candidates': [
                {
                    'sharpe': float(c['sharpe']),
                    'max_drawdown': float(c['max_drawdown']),
                    'turnover': int(c['turnover']),
                    'policy': c['policy']
                } for c in top_candidates[:8]
            ]
        }
        refine_messages = [
            {"role": "system", "content": "Revise rules step-by-step, but output only a valid JSON array of policies."},
            {"role": "user", "content": json.dumps(refine_payload)}
        ]

        try:
            # Some models (e.g., 'o3-mini') do not support 'temperature'
            kwargs = {
                'model': OPENAI_MODEL,
                'messages': refine_messages,
                'max_completion_tokens': REFINEMENT_MAX_TOKENS,
            }
            if not str(OPENAI_MODEL).lower().startswith('o3'):
                kwargs['temperature'] = 0.3
            res = client.chat.completions.create(**kwargs)
            content = res.choices[0].message.content
            # Try strict parse; if fails, attempt bracket slice
            try:
                raw_refined = json.loads(content)
                if not isinstance(raw_refined, list):
                    raw_refined = []
            except Exception:
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1 and end > start:
                    try:
                        raw_refined = json.loads(content[start:end+1])
                        if not isinstance(raw_refined, list):
                            raw_refined = []
                    except Exception:
                        raw_refined = []
                else:
                    raw_refined = []
        except Exception as e:
            print(f"⚠️  Refinement round {r} failed: {e}")
            break

        refined_policies = validate_and_prepare_policies(raw_refined, feature_stats)

        # Additionally, create local mutations of current top policies to broaden search
        def mutate_policy(pol: Dict[str, Any]) -> Dict[str, Any]:
            rng = np.random.default_rng()
            mutated = json.loads(json.dumps(pol))
            if mutated.get('rules'):
                r_idx = int(rng.integers(low=0, high=len(mutated['rules'])))
                rule = mutated['rules'][r_idx]
                if rule.get('if'):
                    p_idx = int(rng.integers(low=0, high=len(rule['if'])))
                    pred = rule['if'][p_idx]
                    feat = pred.get('feat')
                    if feat in feature_stats:
                        qs = list(feature_stats[feat]['quantiles'].items())
                        qs_sorted = sorted(qs, key=lambda x: float(x[0]))
                        # neighbor quantile threshold
                        if 'thr_q' in pred:
                            curr_q = float(pred['thr_q'])
                            q_vals = [float(k) for k, _ in qs_sorted]
                            nearest = min(range(len(q_vals)), key=lambda i: abs(q_vals[i] - curr_q))
                            new_idx = min(len(q_vals)-1, max(0, nearest + int(rng.integers(-1, 2))))
                            pred['thr_q'] = q_vals[new_idx]
                        else:
                            new_q = float(rng.choice([0.2,0.3,0.5,0.7,0.8]))
                            pred['thr_q'] = new_q
                            pred.pop('thr', None)
            return mutated

        local_mutations = [mutate_policy(c['policy']) for c in top_candidates for _ in range(6)]
        refined_policies.extend(validate_and_prepare_policies(local_mutations, feature_stats))

        # Evaluate refined policies with hard balance screen
        if len(refined_policies) == 0:
            print("No valid refined policies returned; stopping.")
            break

        round_results: List[Dict[str, Any]] = []
        filtered_balance_ref = 0
        tested_ref = 0
        for p in refined_policies:
            res = evaluate_policy_objective(p, internal_train, feature_cols)
            share = res.get('pred_up_share', 0.5)
            tested_ref += 1
            if share < PRED_UP_HARD_MIN or share > PRED_UP_HARD_MAX:
                filtered_balance_ref += 1
                continue
            round_results.append(res)
        print(f"Refinement round {r+1}: tested {tested_ref}, filtered by balance: {filtered_balance_ref}, retained: {len(round_results)}")

        # Combine and select top M
        combined = top_candidates + round_results
        top_candidates = select_top(combined, TOP_M)

        # Save round logs
        with open(f'policies_round_{r+1}_results.json', 'w') as f:
            json.dump(top_candidates, f, indent=2)

        # Early stop if no improvement
        best_sharpe = max(c['sharpe'] for c in top_candidates) if top_candidates else -1e9
        if r > 0:
            prev_best = max(c['sharpe'] for c in current_candidates_results) if 'current_candidates_results' in locals() else -1e9
            if best_sharpe <= prev_best:
                print("Early stop: no Sharpe improvement.")
                break
        current_candidates_results = top_candidates

    # Selection on VALID (robust rolling aggregation)
    print("\n🔎 Selecting final policy on VALID using rolling VALID aggregation...")
    final_set = [c['policy'] for c in top_candidates]
    if not final_set:
        print("⚠️  No candidate policies available. Exiting.")
        return {}

    valid_results: List[Dict[str, Any]] = []
    tested_valid = 0
    filtered_balance_valid = 0
    for p in final_set:
        agg = evaluate_over_rolling_valid(p, pd.concat([internal_train, valid], ignore_index=True), feature_cols, k_folds=3)
        # Also compute straight VALID metrics for reporting
        straight = evaluate_policy_objective(p, valid, feature_cols)
        # Apply hard balance on VALID too
        share_valid = straight.get('pred_up_share', 0.5)
        tested_valid += 1
        if share_valid < PRED_UP_HARD_MIN or share_valid > PRED_UP_HARD_MAX:
            filtered_balance_valid += 1
            continue
        valid_results.append({
            'policy': p,
            'avg_sharpe': agg['avg_sharpe'],
            'avg_maxdd': agg['avg_maxdd'],
            'avg_turnover': agg['avg_turnover'],
            'valid_sharpe': straight['sharpe'],
            'valid_maxdd': straight['max_drawdown'],
            'valid_turnover': straight['turnover']
        })
    print(f"VALID selection: tested {tested_valid}, filtered by balance: {filtered_balance_valid}, retained: {len(valid_results)}")

    if not valid_results:
        print("⚠️  All candidates failed strict prediction balance (47–53%) on VALID. Regenerating candidates...")
        return reasoning_policy_search(internal_train, valid, feature_cols)

    # Select by avg_sharpe desc, then avg_maxdd asc, then avg_turnover asc, then closeness to 50% UP on VALID
    def closeness_to_50(policy: Dict[str, Any]) -> float:
        Xv = valid[feature_cols]
        preds_v = evaluate_decision_list(policy, Xv)
        share = float((preds_v == 1).sum()) / len(preds_v)
        return abs(share - 0.5)
    valid_sorted = sorted(
        valid_results,
        key=lambda r: (
            -r['avg_sharpe'],
            r['avg_maxdd'],
            r['avg_turnover'],
            closeness_to_50(r['policy'])
        )
    )
    best_on_valid = {
        'policy': valid_sorted[0]['policy'],
        'sharpe': valid_sorted[0]['valid_sharpe'],
        'max_drawdown': valid_sorted[0]['valid_maxdd'],
        'turnover': valid_sorted[0]['valid_turnover']
    }

    # Freeze policy
    final_policy = best_on_valid['policy']
    with open('final_policy.json', 'w') as f:
        json.dump(final_policy, f, indent=2)
    print("✅ Final policy saved to final_policy.json")

    # Update persistent policy bank
    try:
        bank = []
        if os.path.exists(POLICY_BANK_PATH):
            with open(POLICY_BANK_PATH, 'r') as f:
                try:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        bank = loaded
                    else:
                        bank = []
                except Exception:
                    bank = []
        bank.append({
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'avg_sharpe': valid_sorted[0]['avg_sharpe'] if 'valid_sorted' in locals() else None,
            'avg_maxdd': valid_sorted[0]['avg_maxdd'] if 'valid_sorted' in locals() else None,
            'avg_turnover': valid_sorted[0]['avg_turnover'] if 'valid_sorted' in locals() else None,
            'valid_sharpe': best_on_valid['sharpe'],
            'valid_maxdd': best_on_valid['max_drawdown'],
            'valid_turnover': best_on_valid['turnover'],
            'policy': final_policy
        })
        # Deduplicate identical policies and keep top by avg_sharpe
        unique = {}
        for entry in bank:
            key = json.dumps(entry['policy'], sort_keys=True)
            if key not in unique or (entry.get('avg_sharpe') or -1e9) > (unique[key].get('avg_sharpe') or -1e9):
                unique[key] = entry
        ranked = sorted(unique.values(), key=lambda e: -(e.get('avg_sharpe') or -1e9))
        best_all_time = ranked[:POLICY_BANK_SIZE]
        with open(POLICY_BANK_PATH, 'w') as f:
            json.dump(best_all_time, f, indent=2)
        print(f"💾 Policy bank updated at {POLICY_BANK_PATH} (kept {min(len(ranked), POLICY_BANK_SIZE)} entries)")
        # Print comparison header (no policy content)
        if best_all_time:
            best_entry = best_all_time[0]
            print("\n🏅 Best-of-All-Time on VALID (rolling avg):")
            print(f"  avg Sharpe: {best_entry.get('avg_sharpe')} | avg MaxDD: {best_entry.get('avg_maxdd')} | avg Turnover: {best_entry.get('avg_turnover')}")
    except Exception as e:
        print(f"⚠️  Failed to update policy bank: {e}")

    return best_on_valid


# =============================
# MAIN
# =============================

def main() -> None:
    # Load data and features
    df, feature_cols = load_and_prepare_data('feature_matrix.csv')

    # Target distribution overview
    print("\n📈 Target distribution:")
    y_all = df['target_next_day']
    target_counts = y_all.value_counts().sort_index()
    label_map = {-1: 'Down (<0%)', 1: 'Up (>=0%)'}
    for t, c in target_counts.items():
        pct = c / len(y_all) * 100
        print(f"  {label_map.get(t, t)}: {c} ({pct:.1f}%)")

    # Chronological split 70/30
    train_df, test_df = chronological_split(df, train_ratio=0.7)

    # Internal split for search
    internal_train, valid = internal_train_valid_split(train_df)

    # Reasoning policy search and selection
    best_on_valid = reasoning_policy_search(internal_train, valid, feature_cols)
    if not best_on_valid:
        print("⚠️  No policy found. Exiting.")
        return

    final_policy = best_on_valid['policy']
    print("\n📝 Final policy (frozen):")
    print(json.dumps(final_policy, indent=2))

    # One-shot evaluation on TEST (full metrics & simulation)
    print("\n📦 Evaluating frozen policy on TEST set...")
    X_test = test_df[feature_cols]
    y_test = test_df['target_next_day'].values
    preds_test = evaluate_decision_list(final_policy, X_test)

    # Classification and bias analysis
    metrics = classification_and_bias_analysis(y_test, preds_test)
    cm = metrics['confusion_matrix']  # [[TN-like for -1], [FP-like]] but we will format

    print(f"\n📈 Detailed Performance:")
    print(f"  Down    : Accuracy={metrics['down_accuracy']:.3f}")
    print(f"  Up      : Accuracy={metrics['up_accuracy']:.3f}")

    print(f"\n🔄 Confusion Matrix:")
    print(f"       Predicted")
    print(f"     Down   Up")
    print(f"     ────  ────")
    # cm is 2x2 with order [-1, 1]
    print(f"Down {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Up   {cm[1][0]:4d}  {cm[1][1]:4d}")

    binary_accuracy = metrics['accuracy']
    print(f"\n🎯 Binary Classification Accuracy: {binary_accuracy:.3f} ({binary_accuracy:.1%})")

    print(f"\n📊 Class-specific Performance:")
    print(f"  When actual was Down: {metrics['down_accuracy']:.1%} predicted correctly")
    print(f"  When actual was Up  : {metrics['up_accuracy']:.1%} predicted correctly")

    print(f"\n" + "=" * 70)
    print(f"🎯 UP BIAS ANALYSIS")
    print(f"=" * 70)
    test_total = len(y_test)
    test_actual_up = int((y_test == 1).sum())
    test_actual_down = int((y_test == -1).sum())
    test_pred_up = int((preds_test == 1).sum())
    test_pred_down = int((preds_test == -1).sum())
    up_mask = (y_test == 1)
    down_mask = (y_test == -1)
    up_acc = float((preds_test[up_mask] == 1).mean()) if up_mask.sum() > 0 else 0.0
    down_acc = float((preds_test[down_mask] == -1).mean()) if down_mask.sum() > 0 else 0.0

    print(f"📊 Test Period Market Distribution:")
    print(f"  Actual UP days: {test_actual_up} ({test_actual_up/test_total*100:.1f}%)")
    print(f"  Actual DOWN days: {test_actual_down} ({test_actual_down/test_total*100:.1f}%)")
    print(f"\n🤖 Model Prediction Distribution:")
    print(f"  Predicted UP: {test_pred_up} ({test_pred_up/test_total*100:.1f}%)")
    print(f"  Predicted DOWN: {test_pred_down} ({test_pred_down/test_total*100:.1f}%)")
    print(f"\n📈 Class-Specific Accuracy:")
    print(f"  UP accuracy: {up_acc:.1%} ({int((preds_test[up_mask] == 1).sum())}/{int(up_mask.sum())} correct)")
    print(f"  DOWN accuracy: {down_acc:.1%} ({int((preds_test[down_mask] == -1).sum())}/{int(down_mask.sum())} correct)")

    market_up_bias = test_actual_up / test_total if test_total > 0 else 0.5
    model_up_bias = test_pred_up / test_total if test_total > 0 else 0.5
    print(f"\n💡 UP Bias Analysis:")
    if abs(market_up_bias - 0.5) > abs(model_up_bias - 0.5):
        print(f"  🎯 MARKET DRIVEN: Test period was {'UP' if market_up_bias > 0.5 else 'DOWN'} biased ({market_up_bias:.1%} UP)")
        print(f"      Model predictions are more balanced ({model_up_bias:.1%} UP)")
        print(f"      Good UP performance likely due to favorable market conditions")
    elif model_up_bias > market_up_bias + 0.05:
        print(f"  ⚠️  MODEL BIAS: Model over-predicts UP ({model_up_bias:.1%} vs {market_up_bias:.1%} actual)")
        print(f"      Model has learned a systematic UP bias")
    elif model_up_bias < market_up_bias - 0.05:
        print(f"  ⚠️  MODEL BIAS: Model under-predicts UP ({model_up_bias:.1%} vs {market_up_bias:.1%} actual)")
        print(f"      Model has learned a systematic DOWN bias")
    else:
        print(f"  ✅ BALANCED: Model predictions align with market distribution")
        print(f"      UP performance reflects genuine predictive skill")

    # Announce a single TEST simulation once
    print(f"\n" + "=" * 70)
    print(f"💰 SIMPLE TRADING SIMULATION (TEST ONLY)")
    print(f"=" * 70)
    print(f"📊 Simulation Setup:")
    print(f"  Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"  Trading days: {len(test_df)}")
    print(f"  Initial capital: $10,000")
    print(f"  Strategy: Long when predict UP (+1), Short when predict DOWN (-1)")
    if ENABLE_TRANSACTION_COSTS:
        print(f"  Transaction costs: {0.2:.1f}% per round trip (0.1% buy + 0.1% sell)")
    else:
        print(f"  Transaction costs: DISABLED (0.0%)")

    # Run simulation on TEST and output CSV (reasoning-specific file)
    sim_results = run_simulation(
        dates=test_df['date'].values,
        predictions=preds_test,
        actual_returns_pct=test_df['sol_actual_next_day_return'].values,
        actual_targets=y_test,
        output_csv_path='daily_trading_results_reasoning.csv',
    )

    # Also compute performance for best-of-all-time policy from bank (if any)
    best_all_time_entry = None
    if os.path.exists(POLICY_BANK_PATH):
        try:
            with open(POLICY_BANK_PATH, 'r') as f:
                bank = json.load(f)
            if isinstance(bank, list) and len(bank) > 0 and isinstance(bank[0], dict) and 'policy' in bank[0]:
                best_all_time_entry = bank[0]
        except Exception:
            best_all_time_entry = None

    best_all_time_sim = None
    if best_all_time_entry is not None:
        best_policy = best_all_time_entry['policy']
        preds_best = evaluate_decision_list(best_policy, X_test)
        best_all_time_sim = run_simulation(
            dates=test_df['date'].values,
            predictions=preds_best,
            actual_returns_pct=test_df['sol_actual_next_day_return'].values,
            actual_targets=y_test,
            output_csv_path=None,
        )

    # Consolidated Simulation Results (verbatim style)
    print(f"\n📊 COMPREHENSIVE REASONING STRATEGY RESULTS:")
    print(f"=" * 70)
    print(f"🧠 Reasoning Long/Short Strategy:")
    print(f"  Final Portfolio Value: ${sim_results['ml_capital']:,.2f}")
    print(f"  Total Return: {sim_results['ml_total_return']:.1%}")
    ml_daily_returns = []  # compute from detailed log to print daily mean/vol
    bh_daily_returns = []
    log = sim_results['detailed_log']
    for i in range(len(log)):
        if i == 0:
            ml_prev = log[i]['ml_balance_before']
            bh_prev = log[i]['buy_hold_balance_before']
        else:
            ml_prev = log[i-1]['ml_balance_after']
            bh_prev = log[i-1]['buy_hold_balance_after']
        ml_curr = log[i]['ml_balance_after']
        bh_curr = log[i]['buy_hold_balance_after']
        ml_daily_returns.append((ml_curr - ml_prev) / ml_prev)
        bh_daily_returns.append((bh_curr - bh_prev) / bh_prev)
    ml_mean = float(np.mean(ml_daily_returns)) if ml_daily_returns else 0.0
    ml_std = float(np.std(ml_daily_returns, ddof=1)) if len(ml_daily_returns) > 1 else 0.0
    bh_mean = float(np.mean(bh_daily_returns)) if bh_daily_returns else 0.0
    bh_std = float(np.std(bh_daily_returns, ddof=1)) if len(bh_daily_returns) > 1 else 0.0

    print(f"  Daily Mean Return: {ml_mean:.4f} ({ml_mean*100:.2f}%)")
    print(f"  Daily Volatility: {ml_std:.4f} ({ml_std*100:.2f}%)")
    print(f"  Sharpe Ratio: {sim_results['ml_sharpe_annual']:.3f}")
    print(f"  Max Drawdown: {sim_results['ml_max_drawdown']:.1%}")
    print(f"  Win Rate: {sim_results['win_rate']:.1%} ({sim_results['winning_days']}/{sim_results['total_days']} days)")

    print(f"\n📊 Buy & Hold Strategy:")
    print(f"  Final Portfolio Value: ${sim_results['buy_hold_capital']:,.2f}")
    print(f"  Total Return: {sim_results['buy_hold_total_return']:.1%}")
    print(f"  Daily Mean Return: {bh_mean:.4f} ({bh_mean*100:.2f}%)")
    print(f"  Daily Volatility: {bh_std:.4f} ({bh_std*100:.2f}%)")
    print(f"  Sharpe Ratio: {sim_results['buy_hold_sharpe_annual']:.3f}")
    print(f"  Max Drawdown: {sim_results['buy_hold_max_drawdown']:.1%}")

    print(f"\n🏆 STRATEGY COMPARISON (Reasoning vs Baseline):")
    print(f"=" * 70)
    print(f"{'Strategy':<20} {'Final Value':<12} {'Return':<10} {'Sharpe':<8} {'vs Buy&Hold':<12}")
    print(f"-" * 70)
    print(f"{'Reasoning L/S (this run)':<20} ${sim_results['ml_capital']:<11,.0f} {sim_results['ml_total_return']:>8.1%} {sim_results['ml_sharpe_annual']:>6.2f} {(sim_results['ml_total_return'] - sim_results['buy_hold_total_return'])*100:>10.1f}%")
    if best_all_time_sim is not None:
        print(f"{'Reasoning L/S (best all-time)':<20} ${best_all_time_sim['ml_capital']:<11,.0f} {best_all_time_sim['ml_total_return']:>8.1%} {best_all_time_sim['ml_sharpe_annual']:>6.2f} {(best_all_time_sim['ml_total_return'] - sim_results['buy_hold_total_return'])*100:>10.1f}%")
    print(f"{'Buy & Hold':<20} ${sim_results['buy_hold_capital']:<11,.0f} {sim_results['buy_hold_total_return']:>8.1%} {sim_results['buy_hold_sharpe_annual']:>6.2f} {'0.0%':>10}")

    total_trades = len([d for d in log if d['prediction'] != 0])
    print(f"\n💸 TRANSACTION COST IMPACT:")
    if ENABLE_TRANSACTION_COSTS:
        total_cost_pct = total_trades * (0.002) * 100
        print(f"  Total trades: {total_trades} | Cost per trade: {0.2:.1f}% | Total cost: {total_cost_pct:.1f}%")
    else:
        print(f"  Transaction costs: DISABLED (0.0%) - Running simulation without costs")

    print(f"\n✅ Complete trading data saved to: daily_trading_results_reasoning.csv")
    print(f"🎉 Simulation completed - {len(log)} trading days analyzed")


if __name__ == "__main__":
    main()

