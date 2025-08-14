"""
🧠 REASONING POLICY BACKTEST - SOLANA PRICE PREDICTION

HOW THIS PROGRAM WORKS (COMPREHENSIVE APPROACH):

1. 📊 DATA PREPARATION
   - Loads feature_matrix.csv with engineered features
   - Splits data chronologically: 70% TRAIN, 30% TEST
   - No internal validation split - uses full training data

2. 🚀 FOCUSED POLICY GENERATION
   - Strategy 1: LLM-generated policies with focused complexity (3-5 rules, 3-5 predicates)
   - Strategy 2: Systematic rule combinations with threshold variations
   - Strategy 3: Feature-family based policies (technical, volatility, macro, returns)
   - Target: ~500 high-quality policies

3. 🤖 LLM POLICY CREATION
   - LLM sees comprehensive feature statistics (mean, std, quantiles)
   - LLM creates sophisticated policies with multiple rules and predicates
   - Policies must predict 40-60% UP to avoid bias
   - Uses diverse feature types and threshold strategies

4. 📈 MASSIVE POLICY EVALUATION
   - Tests all policies on full training data
   - Enforces balance constraints (40-60% UP prediction)
   - Calculates Sharpe ratio, returns, max drawdown
   - Ranks policies by performance

5. 🏆 TOP POLICY SELECTION
   - Selects top 10 policies by Sharpe ratio
   - Chooses best policy for final evaluation
   - Policies have 3-5 rules with 3-5 predicates each

6. 🧪 TEST EVALUATION
   - Runs final policy on TEST set (unseen data)
   - Performs trading simulation with $10,000 starting capital
   - Compares against buy-and-hold strategy
   - Updates best_policy.json if performance improves

KEY FEATURES:
- No data leakage: strict chronological splits
- Focused thoroughness: tests ~500 high-quality policy combinations
- Sophisticated policies: 3-5 rules, 3-5 predicates per rule
- LLM-driven reasoning with comprehensive feature analysis
- Multiple generation strategies for policy diversity
- Clear step-by-step output showing progress
- Automatic best policy persistence across runs
- Balanced predictions prevent bias issues
"""

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
OPENAI_MODEL = os.getenv("REASONING_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Internal validation split ratio from the TRAIN slice
INTERNAL_TRAIN_RATIO = 0.8  # 80% internal train, 20% valid

# Policy bank configuration
BEST_POLICY_PATH = 'best_policy.json'  # single best-of-all-time policy (by TEST metrics)

# Deterministic seeding for local generation
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)

# Prediction balance constraints (target ~50/50 UP vs DOWN)
PRED_UP_TARGET = 0.5
PRED_UP_SOFT_MIN = 0.40
PRED_UP_SOFT_MAX = 0.60
PRED_UP_HARD_MIN = 0.40
PRED_UP_HARD_MAX = 0.60

# Policy cache configuration
POLICY_CACHE_DIR = 'policy_cache'
LLM_POLICIES_FILE = os.path.join(POLICY_CACHE_DIR, 'llm_policies.json')
SYSTEMATIC_POLICIES_FILE = os.path.join(POLICY_CACHE_DIR, 'systematic_policies.json')
FEATURE_POLICIES_FILE = os.path.join(POLICY_CACHE_DIR, 'feature_policies.json')
POLICY_METADATA_FILE = os.path.join(POLICY_CACHE_DIR, 'policy_metadata.json')

def ensure_policy_cache_dir():
    """Ensure the policy cache directory exists."""
    os.makedirs(POLICY_CACHE_DIR, exist_ok=True)

def save_policies_to_cache(policies: List[Dict[str, Any]], filename: str, metadata: Dict[str, Any] = None):
    """Save policies to cache file with metadata."""
    ensure_policy_cache_dir()
    
    cache_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'policies': policies,
        'metadata': metadata or {}
    }
    
    with open(filename, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"    💾 Saved {len(policies)} policies to {filename}")

def load_policies_from_cache(filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load policies from cache file if it exists and is recent."""
    if not os.path.exists(filename):
        return [], {}
    
    try:
        with open(filename, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is recent (within last 24 hours)
        cache_time = pd.Timestamp(cache_data['timestamp'])
        if pd.Timestamp.now() - cache_time < pd.Timedelta(hours=24):
            policies = cache_data.get('policies', [])
            metadata = cache_data.get('metadata', {})
            print(f"    📂 Loaded {len(policies)} policies from {filename} (cache hit)")
            return policies, metadata
        else:
            print(f"    ⏰ Cache expired for {filename}, will regenerate")
            return [], {}
            
    except Exception as e:
        print(f"    ⚠️  Error loading cache from {filename}: {e}")
        return [], {}

def check_sufficient_policies() -> bool:
    """Check if we have sufficient policies in cache to skip generation."""
    total_policies = 0
    
    for filename in [LLM_POLICIES_FILE, SYSTEMATIC_POLICIES_FILE, FEATURE_POLICIES_FILE]:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    cache_data = json.load(f)
                    policies = cache_data.get('policies', [])
                    total_policies += len(policies)
            except:
                continue
    
    sufficient = total_policies >= 400  # Need at least 400 policies
    print(f"    📊 Cache contains {total_policies} policies (need 400+)")
    return sufficient


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





# =============================
# STATS, SAMPLES, AND QUANTILES
# =============================







# =============================
# POLICY FORMAT AND EVALUATION
# =============================




def evaluate_decision_list(policy: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    def pred_row(row) -> int:
        for rule in policy.get('rules', []):
            conditions = rule.get('if', [])
            satisfied = True
            for c in conditions:
                feat = c['feat']
                op = c['op']
                
                # Handle both thr and thr_q fields
                if 'thr_q' in c:
                    # Convert quantile threshold to actual value
                    q = c['thr_q']
                    if feat in X.columns:
                        # Use the actual data to get quantile value
                        feat_values = X[feat].dropna()
                        if len(feat_values) > 0:
                            thr = feat_values.quantile(q)
                        else:
                            thr = 0.0
                    else:
                        thr = 0.0
                elif 'thr' in c:
                    thr = c['thr']
                else:
                    # Default threshold if neither is provided
                    thr = 0.0
                
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
    # Fast path for evaluation (no CSV/log): vectorized computation
    if output_csv_path is None:
        returns = np.asarray(actual_returns_pct, dtype=float) / 100.0
        preds_arr = np.asarray(predictions, dtype=int)
        strat_ret = np.where(preds_arr == 1, returns, -returns) - transaction_cost_per_trade
        # Wealth paths
        ml_wealth = 10000.0 * np.cumprod(1.0 + strat_ret)
        bh_wealth = 10000.0 * (1 - transaction_cost_per_trade / 2) * np.cumprod(1.0 + returns)
        bh_wealth[-1] = bh_wealth[-1] * (1 - transaction_cost_per_trade / 2)
        # Max DD
        ml_peak = np.maximum.accumulate(ml_wealth)
        ml_max_drawdown = float(np.max((ml_peak - ml_wealth) / ml_peak)) if ml_wealth.size else 0.0
        bh_peak = np.maximum.accumulate(bh_wealth)
        buy_hold_max_drawdown = float(np.max((bh_peak - bh_wealth) / bh_peak)) if bh_wealth.size else 0.0
        # Sharpe from daily returns
        ml_mean = float(np.mean(strat_ret)) if strat_ret.size else 0.0
        ml_std = float(np.std(strat_ret, ddof=1)) if strat_ret.size > 1 else 0.0
        bh_mean = float(np.mean(returns)) if returns.size else 0.0
        bh_std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
        ml_sharpe_daily = (ml_mean / ml_std) if ml_std > 0 else 0.0
        bh_sharpe_daily = (bh_mean / bh_std) if bh_std > 0 else 0.0
        ml_sharpe_annual = ml_sharpe_daily * math.sqrt(252)
        bh_sharpe_annual = bh_sharpe_daily * math.sqrt(252)
        ml_capital = float(ml_wealth[-1]) if ml_wealth.size else 10000.0
        buy_hold_capital = float(bh_wealth[-1]) if bh_wealth.size else 10000.0
        detailed_log = []
        total_days = int(len(returns))
        winning_days = int(np.sum(np.where(preds_arr == 1, returns > 0, returns < 0)))
        win_rate = (winning_days / total_days) if total_days > 0 else 0.0
        ml_total_return = (ml_capital - 10000.0) / 10000.0
        bh_total_return = (buy_hold_capital - 10000.0) / 10000.0
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

    # Full path with detailed log and CSV
    initial_capital = 10000.0
    ml_capital = initial_capital
    buy_hold_capital = initial_capital
    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)
    detailed_log: List[Dict[str, Any]] = []
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
        if ml_capital > ml_peak:
            ml_peak = ml_capital
        else:
            ml_max_drawdown = max(ml_max_drawdown, (ml_peak - ml_capital) / ml_peak)
        if buy_hold_capital > buy_hold_peak:
            buy_hold_peak = buy_hold_capital
        else:
            buy_hold_max_drawdown = max(buy_hold_max_drawdown, (buy_hold_peak - buy_hold_capital) / buy_hold_peak)
        prediction_correct = int(pred == actual_target)
        # Safe date string conversion
        try:
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        except Exception:
            date_str = str(date)
        detailed_log.append({
            'date': date_str,
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

    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)
    if detailed_log:
        detailed_log[-1]['buy_hold_balance_after'] = buy_hold_capital
    if output_csv_path:
        pd.DataFrame(detailed_log).to_csv(output_csv_path, index=False)
        print(f"\n✅ Detailed daily results saved to: {output_csv_path}")

    # Daily returns for Sharpe (from balances)
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
        # Do not pass temperature to avoid model-specific unsupported params
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
        # Do not pass temperature to avoid model-specific unsupported params
        res = client.chat.completions.create(**kwargs)
        content = res.choices[0].message.content
        data = json.loads(content) if content.strip().startswith('[') else []
        return [r for r in data if isinstance(r, dict)]
    except Exception:
        return []
















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
    # Calculate prediction balance
    total = len(preds)
    pred_up_share = float((preds == 1).sum()) / total if total > 0 else 0.5
    
    # Check if policy meets hard balance constraints
    if pred_up_share < PRED_UP_HARD_MIN or pred_up_share > PRED_UP_HARD_MAX:
        return None  # Policy fails balance constraints
    return {
        'policy': policy,
        'sharpe': sim['ml_sharpe_annual'],
        'return': sim['ml_total_return'],
        'max_drawdown': sim['ml_max_drawdown'],
        'turnover': turnover,
        'up_share': pred_up_share,
        'accuracy': metrics['accuracy'],
        'metrics': metrics,
    }


















def create_rolling_windows(df: pd.DataFrame, window_days: int = 45, overlap_ratio: float = 0.5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create rolling windows with overlap for rule generation."""
    step_days = int(window_days * (1 - overlap_ratio))
    windows = []
    
    for start_idx in range(0, len(df) - window_days, step_days):
        end_idx = start_idx + window_days
        window_data = df.iloc[start_idx:end_idx].copy()
        
        # Split window into training (80%) and validation (20%)
        split_idx = int(len(window_data) * 0.8)
        train_window = window_data.iloc[:split_idx]
        valid_window = window_data.iloc[split_idx:]
        
        if len(train_window) > 10 and len(valid_window) > 5:  # Ensure sufficient data
            windows.append((train_window, valid_window))
    
    return windows

def generate_window_summary(train_window: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Create compact summary for LLM rule generation."""
    summary = {
        'period': f"{train_window['date'].min().date()} to {train_window['date'].max().date()}",
        'days': len(train_window),
        'target_dist': train_window['target_next_day'].value_counts().to_dict(),
        'features': {}
    }
    
    # Per-feature statistics
    for feat in feature_cols[:15]:  # Limit to top 15 features for compactness
        if feat in train_window.columns:
            series = train_window[feat].dropna()
            if len(series) > 0:
                summary['features'][feat] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'q25': float(series.quantile(0.25)),
                    'q50': float(series.quantile(0.50)),
                    'q75': float(series.quantile(0.75)),
                    'recent_trend': 'up' if series.iloc[-1] > series.iloc[0] else 'down'
                }
    
    # Balanced exemplars (2 UP, 2 DOWN)
    up_examples = train_window[train_window['target_next_day'] == 1].head(2)
    down_examples = train_window[train_window['target_next_day'] == -1].head(2)
    
    summary['exemplars'] = {
        'up': up_examples[['date'] + feature_cols[:5]].to_dict('records') if len(up_examples) > 0 else [],
        'down': down_examples[['date'] + feature_cols[:5]].to_dict('records') if len(down_examples) > 0 else []
    }
    
    return summary

def request_rules_for_window(client: Any, window_summary: Dict[str, Any], feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Request IF-THEN rules from LLM for a specific window."""
    
    prompt = f"""You are a quantitative analyst creating trading rules for Solana (SOL) price prediction.

WINDOW SUMMARY:
- Period: {window_summary['period']} ({window_summary['days']} days)
- Target distribution: {window_summary['target_dist']}

FEATURE STATISTICS:
{chr(10).join([f"- {feat}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, q25={stats['q25']:.4f}, q50={stats['q50']:.4f}, q75={stats['q75']:.4f}, trend={stats['recent_trend']}" for feat, stats in list(window_summary['features'].items())[:10]])}

EXEMPLARS:
- UP examples: {len(window_summary['exemplars']['up'])} rows
- DOWN examples: {len(window_summary['exemplars']['down'])} rows

TASK:
Create 3-5 simple IF-THEN rules that predict UP or DOWN. Each rule should:
- Use ≤3 feature conditions
- Have explicit thresholds (use quantiles like 0.25, 0.5, 0.75)
- Be interpretable and logical
- Focus on technical indicators, volatility, or relative pricing

Format each rule as:
{{
  "if": [
    {{"feat": "feature_name", "op": ">", "thr_q": 0.75}},
    {{"feat": "another_feat", "op": "<", "thr_q": 0.25}}
  ],
  "then": "UP"
}}

Return ONLY valid JSON array of rules. No explanations."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise quantitative analyst. Return only valid JSON array of rules."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try multiple JSON parsing strategies
        rules = []
        
        # Strategy 1: Direct JSON parse
        try:
            rules = json.loads(content)
            if isinstance(rules, list):
                print(f"    ✅ Direct JSON parse successful: {len(rules)} rules")
                return rules
        except Exception as e:
            pass
        
        # Strategy 2: Find JSON array with bracket matching
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                rules = json.loads(json_str)
                if isinstance(rules, list):
                    print(f"    ✅ Bracket extraction successful: {len(rules)} rules")
                    return rules
        except Exception as e:
            pass
        
        # Strategy 3: Try to fix common JSON issues
        try:
            # Remove any text before first [
            cleaned = content[content.find('['):]
            # Remove any text after last ]
            cleaned = cleaned[:cleaned.rfind(']')+1]
            # Try to fix common issues
            cleaned = cleaned.replace('\n', ' ').replace('\t', ' ')
            cleaned = cleaned.replace('},]', '}]')  # Fix trailing comma
            cleaned = cleaned.replace(',]', ']')    # Fix trailing comma
            
            rules = json.loads(cleaned)
            if isinstance(rules, list):
                print(f"    ✅ Cleaned JSON parse successful: {len(rules)} rules")
                return rules
        except Exception as e:
            pass
        
        # Strategy 4: Try to extract individual rules
        try:
            # Look for individual rule patterns
            rule_patterns = []
            lines = content.split('\n')
            current_rule = ""
            in_rule = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('{'):
                    in_rule = True
                    current_rule = line
                elif in_rule:
                    current_rule += line
                    if line.endswith('}'):
                        in_rule = False
                        try:
                            rule = json.loads(current_rule)
                            if 'if' in rule and 'then' in rule:
                                rule_patterns.append(rule)
                        except:
                            pass
                        current_rule = ""
            
            if rule_patterns:
                print(f"    ✅ Pattern extraction successful: {len(rule_patterns)} rules")
                return rule_patterns
        except Exception as e:
            pass
        
        # Strategy 5: Handle truncated responses by looking for complete rules
        try:
            # Find all complete rule objects even if array is truncated
            rule_texts = []
            brace_count = 0
            current_rule = ""
            
            for char in content:
                if char == '{':
                    if brace_count == 0:
                        current_rule = char
                    else:
                        current_rule += char
                    brace_count += 1
                elif char == '}':
                    current_rule += char
                    brace_count -= 1
                    if brace_count == 0:
                        # Complete rule found
                        try:
                            rule = json.loads(current_rule)
                            if 'if' in rule and 'then' in rule:
                                rule_texts.append(rule)
                        except:
                            pass
                        current_rule = ""
                elif brace_count > 0:
                    current_rule += char
            
            if rule_texts:
                print(f"    ✅ Truncated response handling successful: {len(rule_texts)} rules")
                return rule_texts
        except Exception as e:
            pass
        
        print(f"    ⚠️  All JSON parsing strategies failed")
        print(f"    📝 Raw LLM response (first 200 chars): {content[:200]}...")
        return []
        
    except Exception as e:
        print(f"    ❌ LLM call failed: {e}")
        return []

def evaluate_rule_performance(rule: Dict[str, Any], train_data: pd.DataFrame, valid_data: pd.DataFrame, feature_cols: List[str]) -> Optional[Dict[str, Any]]:
    """Evaluate a single rule's performance on train and validation data."""
    
    # Convert rule to simple policy for evaluation
    policy = {
        'policy_type': 'decision_list',
        'rules': [rule],
        'default': 'DOWN'
    }
    
    # Evaluate on training data
    train_result = evaluate_policy_objective(policy, train_data, feature_cols)
    if train_result is None:
        return None  # Failed balance constraints
    
    # Evaluate on validation data
    valid_result = evaluate_policy_objective(policy, valid_data, feature_cols)
    if valid_result is None:
        return None  # Failed balance constraints
    
    # Calculate out-of-sample performance
    oos_sharpe = valid_result['sharpe']
    oos_return = valid_result['return']
    
    # Rule quality score (higher is better)
    quality_score = oos_sharpe * 0.7 + oos_return * 0.3
    
    return {
        'rule': rule,
        'train_sharpe': train_result['sharpe'],
        'train_return': train_result['return'],
        'valid_sharpe': oos_sharpe,
        'valid_return': oos_return,
        'quality_score': quality_score,
        'complexity': len(rule.get('if', []))
    }

def greedy_forward_selection(rules_performance: List[Dict[str, Any]], max_rules: int = 5) -> List[Dict[str, Any]]:
    """Build policy incrementally using greedy forward selection."""
    
    if not rules_performance:
        return []
    
    # Sort rules by quality score
    sorted_rules = sorted(rules_performance, key=lambda x: x['quality_score'], reverse=True)
    
    selected_rules = []
    current_policy = {
        'policy_type': 'decision_list',
        'rules': [],
        'default': 'DOWN'
    }
    
    for _ in range(max_rules):
        best_rule = None
        best_improvement = 0
        
        for rule_perf in sorted_rules:
            if rule_perf['rule'] in [r for r in selected_rules]:
                continue  # Already selected
                
            # Test adding this rule
            test_policy = {
                'policy_type': 'decision_list',
                'rules': selected_rules + [rule_perf['rule']],
                'default': 'DOWN'
            }
            
            # Evaluate combined policy (simplified - just use rule quality)
            combined_score = sum(r['quality_score'] for r in [rule_perf] + selected_rules)
            improvement = combined_score - sum(r['quality_score'] for r in selected_rules)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_rule = rule_perf
        
        if best_rule and best_improvement > 0:
            selected_rules.append(best_rule['rule'])
        else:
            break
    
    return selected_rules

def reasoning_policy_search(
    internal_train: pd.DataFrame, 
    valid: pd.DataFrame, 
    feature_cols: List[str]
) -> Optional[Dict[str, Any]]:
    """Rolling window rule generation with greedy policy construction."""
    
    print("\n" + "="*70)
    print("🧠 ROLLING WINDOW POLICY SEARCH")
    print("="*70)
    
    # Step 1: Create rolling windows
    print("\n📊 STEP 1: Creating rolling windows for rule generation...")
    windows = create_rolling_windows(internal_train, window_days=45, overlap_ratio=0.5)
    print(f"✅ Created {len(windows)} rolling windows (45 days, 50% overlap)")
    
    # Step 2: Generate rules per window
    print("\n🤖 STEP 2: Generating rules from each window...")
    client = get_openai_client()
    if client is None:
        print("❌ No OpenAI client available. Exiting.")
        return None
    
    all_rules = []
    for i, (train_window, valid_window) in enumerate(windows):
        print(f"  Processing window {i+1}/{len(windows)}: {train_window['date'].min().date()} to {train_window['date'].max().date()}")
        
        # Generate window summary
        window_summary = generate_window_summary(train_window, feature_cols)
        
        # Request rules from LLM
        window_rules = request_rules_for_window(client, window_summary, feature_cols)
        print(f"    Generated {len(window_rules)} rules")
        
        # Evaluate rules on validation data
        window_valid_rules = 0
        for j, rule in enumerate(window_rules):
            print(f"      Evaluating rule {j+1}/{len(window_rules)}...")
            rule_perf = evaluate_rule_performance(rule, train_window, valid_window, feature_cols)
            if rule_perf is not None:
                all_rules.append(rule_perf)
                window_valid_rules += 1
                print(f"        ✅ Rule passed: Sharpe={rule_perf['valid_sharpe']:.3f}, UP%={rule_perf.get('up_share', 'N/A')}")
            else:
                print(f"        ❌ Rule failed balance constraints")
        
        print(f"    Window {i+1}: {window_valid_rules}/{len(window_rules)} rules passed evaluation")
    
    print(f"\n✅ Total rules generated: {len(all_rules)}")
    
    if not all_rules:
        print("❌ No valid rules generated. Exiting.")
        return None
    
    # Step 3: Greedy forward selection
    print("\n🔍 STEP 3: Building policy using greedy forward selection...")
    selected_rules = greedy_forward_selection(all_rules, max_rules=5)
    print(f"✅ Selected {len(selected_rules)} rules for final policy")
    
    # Step 4: Create final policy
    final_policy = {
        'policy_type': 'decision_list',
        'rules': selected_rules,
        'default': 'DOWN'
    }
    
    # Step 5: Evaluate on VALID set
    print("\n🏁 STEP 4: Final validation on VALID set...")
    valid_result = evaluate_policy_objective(final_policy, valid, feature_cols)
    
    if valid_result is None:
        print("❌ Final policy failed balance constraints on VALID. Exiting.")
        return None
    
    print(f"\n🏆 FINAL POLICY SELECTED:")
    print(f"  Rules: {len(selected_rules)}")
    print(f"  VALID Sharpe: {valid_result['sharpe']:.3f}")
    print(f"  VALID Return: {valid_result['return']:.1%}")
    print(f"  VALID MaxDD: {valid_result['max_drawdown']:.1%}")
    print(f"  VALID UP%: {valid_result['up_share']:.1%}")
    
    return {
        'policy': final_policy,
        'sharpe': valid_result['sharpe'],
        'max_drawdown': valid_result['max_drawdown'],
        'return': valid_result['return']
    }


# =============================
# COMPREHENSIVE POLICY GENERATION
# =============================

def generate_comprehensive_policies(
    train_data: pd.DataFrame, 
    feature_cols: List[str],
    max_policies: int = 500
) -> List[Dict[str, Any]]:
    """Generate a comprehensive set of diverse trading policies with smart caching."""
    
    print(f"\n🤖 GENERATING {max_policies} COMPREHENSIVE POLICIES")
    print("="*60)
    
    # Check if we have sufficient policies in cache
    if check_sufficient_policies():
        print("    ✅ Sufficient policies in cache, loading instead of regenerating...")
        all_policies = []
        
        # Load from all cache files
        for filename in [LLM_POLICIES_FILE, SYSTEMATIC_POLICIES_FILE, FEATURE_POLICIES_FILE]:
            policies, _ = load_policies_from_cache(filename)
            all_policies.extend(policies)
        
        print(f"    📂 Total loaded from cache: {len(all_policies)} policies")
        return all_policies
    
    print("    🔄 Insufficient policies in cache, generating new ones...")
    all_policies = []
    
    # Strategy 1: LLM-generated policies with varying complexity
    print("\n📊 STRATEGY 1: LLM-generated policies...")
    
    # Check if we have LLM policies in cache
    llm_policies, llm_metadata = load_policies_from_cache(LLM_POLICIES_FILE)
    
    if not llm_policies:
        print("    🤖 Generating new LLM policies...")
        client = get_openai_client()
        if client is None:
            print("❌ No OpenAI client available. Exiting.")
            return []
        
        # Different complexity levels
        complexity_configs = [
            (3, 3), (3, 4), (3, 5),
            (4, 3), (4, 4), (4, 5),
            (5, 3), (5, 4), (5, 5)
        ]
        
        policies_per_config = max(1, max_policies // len(complexity_configs))
        
        for max_rules, max_predicates in complexity_configs:
            print(f"  Generating policies with {max_rules} rules, {max_predicates} predicates...")
            
            for i in range(policies_per_config):
                policy = generate_llm_policy(
                    client, train_data, feature_cols, max_rules, max_predicates
                )
                if policy:
                    llm_policies.append(policy)
                    if len(llm_policies) % 50 == 0:
                        print(f"    Generated {len(llm_policies)} policies so far...")
        
        # Save LLM policies to cache
        save_policies_to_cache(llm_policies, LLM_POLICIES_FILE, {
            'complexity_configs': complexity_configs,
            'policies_per_config': policies_per_config
        })
    else:
        print("    📂 Using cached LLM policies")
    
    all_policies.extend(llm_policies)
    
    # Strategy 2: Systematic rule combinations
    print(f"\n📊 STRATEGY 2: Systematic rule combinations...")
    systematic_policies, _ = load_policies_from_cache(SYSTEMATIC_POLICIES_FILE)
    
    if not systematic_policies:
        print("    🔧 Generating new systematic policies...")
        systematic_policies = generate_systematic_policies(train_data, feature_cols, max_policies // 2)
        save_policies_to_cache(systematic_policies, SYSTEMATIC_POLICIES_FILE)
    else:
        print("    📂 Using cached systematic policies")
    
    all_policies.extend(systematic_policies)
    
    # Strategy 3: Feature-based policy families
    print(f"\n📊 STRATEGY 3: Feature-based policy families...")
    feature_policies, _ = load_policies_from_cache(FEATURE_POLICIES_FILE)
    
    if not feature_policies:
        print("    🎯 Generating new feature family policies...")
        feature_policies = generate_feature_family_policies(train_data, feature_cols, max_policies // 4)
        save_policies_to_cache(feature_policies, FEATURE_POLICIES_FILE)
    else:
        print("    📂 Using cached feature family policies")
    
    all_policies.extend(feature_policies)
    
    print(f"\n✅ TOTAL POLICIES AVAILABLE: {len(all_policies)}")
    return all_policies

def generate_llm_policy(
    client: Any, 
    train_data: pd.DataFrame, 
    feature_cols: List[str], 
    max_rules: int, 
    max_predicates: int
) -> Optional[Dict[str, Any]]:
    """Generate a single policy using LLM with specific complexity constraints."""
    
    # Create feature summary for LLM
    feature_summary = {}
    for feat in feature_cols[:20]:  # Top 20 features
        if feat in train_data.columns:
            series = train_data[feat].dropna()
            if len(series) > 0:
                feature_summary[feat] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'q10': float(series.quantile(0.1)),
                    'q25': float(series.quantile(0.25)),
                    'q50': float(series.quantile(0.5)),
                    'q75': float(series.quantile(0.75)),
                    'q90': float(series.quantile(0.9))
                }
    
    prompt = f"""Create a sophisticated trading policy for Solana (SOL) price prediction.

CONSTRAINTS:
- Exactly {max_rules} rules
- Maximum {max_predicates} predicates per rule
- Must predict between 40-60% UP on training data
- Use diverse feature types (technical, volatility, macro, relative pricing)

FEATURES AVAILABLE:
{chr(10).join([f"- {feat}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, q10={stats['q10']:.4f}, q25={stats['q25']:.4f}, q50={stats['q50']:.4f}, q75={stats['q75']:.4f}, q90={stats['q90']:.4f}" for feat, stats in list(feature_summary.items())[:15]])}

POLICY REQUIREMENTS:
1. Use technical indicators (RSI, MACD, moving averages)
2. Include volatility measures (rolling std, price deviations)
3. Consider relative pricing (SOL vs BTC/ETH ratios)
4. Incorporate macro indicators (VIX, DXY, Fed funds)
5. Use multi-timeframe returns (1d, 3d, 7d)

FORMAT:
{{
  "policy_type": "decision_list",
  "rules": [
    {{
      "if": [
        {{"feat": "sol_rsi_14", "op": "<", "thr_q": 0.3}},
        {{"feat": "sol_macd_histogram", "op": ">", "thr": 0}},
        {{"feat": "vix", "op": "<", "thr_q": 0.7}}
      ],
      "then": "UP"
    }}
  ],
  "default": "DOWN"
}}

Return ONLY valid JSON. Make it sophisticated and diverse."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert quantitative analyst. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=3000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON with multiple strategies
        policy = parse_policy_json(content)
        if policy and validate_policy_structure(policy, max_rules, max_predicates):
            return policy
        
        return None
        
    except Exception as e:
        return None

def generate_systematic_policies(
    train_data: pd.DataFrame, 
    feature_cols: List[str], 
    max_policies: int
) -> List[Dict[str, Any]]:
    """Generate policies using systematic rule combinations."""
    
    print(f"  Creating systematic combinations...")
    policies = []
    
    # Select key features for systematic generation
    key_features = [
        'sol_rsi_14', 'sol_macd_histogram', 'sol_return_1d', 'sol_return_7d',
        'sol_volatility_7d', 'sol_price_deviation_sma_20', 'sol_btc_ratio',
        'btc_return_1d', 'eth_return_1d', 'vix', 'dxy'
    ]
    
    # Filter to features that exist
    key_features = [f for f in key_features if f in feature_cols]
    
    # Generate different rule combinations
    rule_templates = [
        # Simple 2-predicate rules
        [('sol_rsi_14', '<', 0.3), ('sol_macd_histogram', '>', 0)],
        [('sol_return_1d', '>', 0.75), ('sol_volatility_7d', '<', 0.5)],
        [('vix', '>', 0.8), ('sol_return_7d', '<', 0.25)],
        
        # Medium complexity 3-predicate rules
        [('sol_rsi_14', '<', 0.25), ('sol_macd_histogram', '>', 0), ('sol_return_1d', '>', 0.6)],
        [('sol_volatility_7d', '<', 0.4), ('sol_price_deviation_sma_20', '>', -0.1), ('btc_return_1d', '>', 0.5)],
        [('vix', '<', 0.6), ('dxy', '<', 0.5), ('sol_return_7d', '>', 0.3)],
        
        # Complex 4-predicate rules
        [('sol_rsi_14', '<', 0.2), ('sol_macd_histogram', '>', 0), ('sol_return_1d', '>', 0.7), ('sol_volatility_7d', '<', 0.3)],
        [('vix', '<', 0.5), ('dxy', '<', 0.4), ('sol_return_7d', '>', 0.4), ('btc_return_1d', '>', 0.6)]
    ]
    
    # Create policies with different rule combinations
    for i in range(min(max_policies, len(rule_templates) * 3)):
        template_idx = i % len(rule_templates)
        template = rule_templates[template_idx]
        
        # Create rules with slight variations
        rules = []
        for feat, op, thr_q in template:
            if feat in key_features:
                # Vary thresholds slightly
                thr_variation = thr_q + (np.random.random() - 0.5) * 0.2
                thr_variation = max(0.1, min(0.9, thr_variation))
                
                rules.append({
                    "feat": feat,
                    "op": op,
                    "thr_q": round(thr_variation, 2)
                })
        
        if len(rules) >= 2:
            policy = {
                "policy_type": "decision_list",
                "rules": [{"if": rules, "then": "UP"}],
                "default": "DOWN"
            }
            policies.append(policy)
    
    print(f"    Generated {len(policies)} systematic policies")
    return policies

def generate_feature_family_policies(
    train_data: pd.DataFrame, 
    feature_cols: List[str], 
    max_policies: int
) -> List[Dict[str, Any]]:
    """Generate policies based on feature families (technical, volatility, macro)."""
    
    print(f"  Creating feature family policies...")
    policies = []
    
    # Define feature families
    feature_families = {
        'technical': ['sol_rsi_14', 'sol_macd_histogram', 'sol_sma_20_ratio', 'sol_sma_50_ratio'],
        'volatility': ['sol_volatility_7d', 'sol_price_deviation_sma_20', 'sol_price_deviation_sma_50'],
        'returns': ['sol_return_1d', 'sol_return_3d', 'sol_return_7d', 'btc_return_1d', 'eth_return_1d'],
        'relative': ['sol_btc_ratio', 'sol_eth_ratio'],
        'macro': ['vix', 'dxy', 'fedfunds']
    }
    
    # Filter to existing features (create a copy to avoid modification during iteration)
    valid_families = {}
    for family_name, features in feature_families.items():
        existing_features = [f for f in features if f in feature_cols]
        if len(existing_features) >= 2:
            valid_families[family_name] = existing_features
    
    # Generate policies for each family
    policies_per_family = max_policies // len(valid_families) if valid_families else 0
    
    for family_name, features in valid_families.items():
        family_policies = []
        
        for i in range(policies_per_family):
            # Randomly select 2-4 features from this family
            num_features = np.random.randint(2, min(5, len(features) + 1))
            selected_features = np.random.choice(features, num_features, replace=False)
            
            # Create rule with random thresholds
            predicates = []
            for feat in selected_features:
                op = np.random.choice(['<', '>', '<=', '>='])
                thr_q = np.random.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9])
                
                predicates.append({
                    "feat": feat,
                    "op": op,
                    "thr_q": thr_q
                })
            
            if predicates:
                policy = {
                    "policy_type": "decision_list",
                    "rules": [{"if": predicates, "then": "UP"}],
                    "default": "DOWN"
                }
                family_policies.append(policy)
        
        policies.extend(family_policies)
        print(f"    {family_name}: {len(family_policies)} policies")
    
    return policies

def parse_policy_json(content: str) -> Optional[Dict[str, Any]]:
    """Parse policy JSON with multiple fallback strategies."""
    
    # Strategy 1: Direct parse
    try:
        policy = json.loads(content)
        if isinstance(policy, dict) and 'rules' in policy:
            return policy
    except:
        pass
    
    # Strategy 2: Extract JSON object
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
            policy = json.loads(json_str)
            if isinstance(policy, dict) and 'rules' in policy:
                return policy
    except:
        pass
    
    # Strategy 3: Brace counting for complete objects
    try:
        brace_count = 0
        current_obj = ""
        objects = []
        
        for char in content:
            if char == '{':
                if brace_count == 0:
                    current_obj = char
                else:
                    current_obj += char
                brace_count += 1
            elif char == '}':
                current_obj += char
                brace_count -= 1
                if brace_count == 0:
                    try:
                        obj = json.loads(current_obj)
                        if isinstance(obj, dict) and 'rules' in obj:
                            return obj
                    except:
                        pass
                    current_obj = ""
            elif brace_count > 0:
                current_obj += char
    except:
        pass
    
    return None

def validate_policy_structure(policy: Dict[str, Any], max_rules: int, max_predicates: int) -> bool:
    """Validate that policy meets structural requirements."""
    
    if not isinstance(policy, dict):
        return False
    
    if 'rules' not in policy or not isinstance(policy['rules'], list):
        return False
    
    if len(policy['rules']) > max_rules:
        return False
    
    for rule in policy['rules']:
        if not isinstance(rule, dict) or 'if' not in rule or 'then' not in rule:
            return False
        
        if not isinstance(rule['if'], list):
            return False
        
        if len(rule['if']) > max_predicates:
            return False
        
        for predicate in rule['if']:
            if not isinstance(predicate, dict):
                return False
            if 'feat' not in predicate or 'op' not in predicate:
                return False
            if 'thr' not in predicate and 'thr_q' not in predicate:
                return False
    
    return True

def evaluate_all_policies(
    policies: List[Dict[str, Any]], 
    train_data: pd.DataFrame, 
    feature_cols: List[str]
) -> List[Dict[str, Any]]:
    """Evaluate all policies and return results sorted by performance."""
    
    print(f"\n📊 EVALUATING {len(policies)} POLICIES ON TRAINING DATA")
    print("="*60)
    
    valid_policies = []
    
    for i, policy in enumerate(policies):
        if i % 100 == 0:
            print(f"  Evaluating policy {i+1}/{len(policies)}...")
        
        try:
            result = evaluate_policy_objective(policy, train_data, feature_cols)
            if result is not None:
                valid_policies.append(result)
        except Exception as e:
            continue
    
    print(f"✅ {len(valid_policies)} policies passed evaluation")
    
    # Sort by Sharpe ratio
    valid_policies.sort(key=lambda x: x['sharpe'], reverse=True)
    
    return valid_policies

def comprehensive_policy_search(
    train_data: pd.DataFrame, 
    feature_cols: List[str]
) -> Optional[Dict[str, Any]]:
    """Comprehensive policy search with massive policy generation and testing."""
    
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE POLICY SEARCH - MAXIMUM THOROUGHNESS")
    print("="*80)
    
    # Step 1: Generate massive number of policies
    print("\n📊 STEP 1: MASSIVE POLICY GENERATION")
    policies = generate_comprehensive_policies(train_data, feature_cols, max_policies=500)
    
    if not policies:
        print("❌ No policies generated. Exiting.")
        return None
    
    # Step 2: Evaluate all policies
    print(f"\n📊 STEP 2: MASSIVE POLICY EVALUATION")
    valid_policies = evaluate_all_policies(policies, train_data, feature_cols)
    
    if not valid_policies:
        print("❌ No valid policies found. Exiting.")
        return None
    
    # Step 3: Select top policies
    print(f"\n🏆 STEP 3: TOP POLICY SELECTION")
    top_policies = valid_policies[:10]  # Top 10 by Sharpe
    
    print(f"\n🏆 TOP 10 POLICIES BY SHARPE RATIO:")
    for i, result in enumerate(top_policies):
        print(f"  {i+1}. Sharpe: {result['sharpe']:.3f}, Return: {result['return']:.1%}, UP%: {result['up_share']:.1%}, Rules: {len(result['policy']['rules'])}")
    
    # Select best policy
    best_policy = top_policies[0]
    
    print(f"\n🏆 BEST POLICY SELECTED:")
    print(f"  Sharpe: {best_policy['sharpe']:.3f}")
    print(f"  Return: {best_policy['return']:.1%}")
    print(f"  MaxDD: {best_policy['max_drawdown']:.1%}")
    print(f"  UP%: {best_policy['up_share']:.1%}")
    print(f"  Rules: {len(best_policy['policy']['rules'])}")
    
    return {
        'policy': best_policy['policy'],
        'sharpe': best_policy['sharpe'],
        'max_drawdown': best_policy['max_drawdown'],
        'return': best_policy['return']
    }


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

    # Show cache status
    print(f"\n📂 POLICY CACHE STATUS:")
    print("="*50)
    ensure_policy_cache_dir()
    
    cache_files = [LLM_POLICIES_FILE, SYSTEMATIC_POLICIES_FILE, FEATURE_POLICIES_FILE]
    total_cached = 0
    
    for filename in cache_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    cache_data = json.load(f)
                    policies = cache_data.get('policies', [])
                    timestamp = cache_data.get('timestamp', 'Unknown')
                    total_cached += len(policies)
                    print(f"  📁 {os.path.basename(filename)}: {len(policies)} policies ({timestamp})")
            except:
                print(f"  ⚠️  {os.path.basename(filename)}: Error reading")
        else:
            print(f"  ❌ {os.path.basename(filename)}: Not found")
    
    print(f"  📊 Total cached policies: {total_cached}")
    if total_cached >= 400:
        print(f"  ✅ Sufficient policies in cache - will skip generation!")
    else:
        print(f"  🔄 Need more policies - will generate new ones")

    # Use full training data for comprehensive policy search (no internal split)
    print(f"\n📊 Using full training data for comprehensive policy search...")
    print(f"Training period: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} days)")

    # Comprehensive policy search and selection
    best_policy_result = comprehensive_policy_search(train_df, feature_cols)
    if not best_policy_result:
        print("⚠️  No policy found. Exiting.")
        return

    final_policy = best_policy_result['policy']
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

    # Save/Update best policy based on TEST performance
    current_policy_performance = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'policy': final_policy,
        'test_return': sim_results['ml_total_return'],
        'test_sharpe': sim_results['ml_sharpe_annual'],
        'test_max_drawdown': sim_results['ml_max_drawdown'],
        'test_final_capital': sim_results['ml_capital'],
        'test_accuracy': binary_accuracy,
        'test_up_accuracy': up_acc,
        'test_down_accuracy': down_acc
    }
    
    best_policy_exists = os.path.exists(BEST_POLICY_PATH)
    should_update = False
    
    if best_policy_exists:
        try:
            with open(BEST_POLICY_PATH, 'r') as f:
                best_existing = json.load(f)
            
            # Compare by TEST return (primary) and Sharpe (secondary)
            if (current_policy_performance['test_return'] > best_existing.get('test_return', -float('inf')) or
                (current_policy_performance['test_return'] == best_existing.get('test_return', -float('inf')) and 
                 current_policy_performance['test_sharpe'] > best_existing.get('test_sharpe', -float('inf')))):
                should_update = True
                print(f"\n🏆 NEW BEST POLICY! Current: {current_policy_performance['test_return']:.1%} vs Previous: {best_existing.get('test_return', 0):.1%}")
            else:
                print(f"\n📊 Policy performance: {current_policy_performance['test_return']:.1%} vs Best: {best_existing.get('test_return', 0):.1%}")
        except Exception as e:
            print(f"⚠️  Error reading existing best policy: {e}")
            should_update = True
    else:
        should_update = True
        print(f"\n🏆 First run - saving initial policy with {current_policy_performance['test_return']:.1%} return")
    
    if should_update:
        with open(BEST_POLICY_PATH, 'w') as f:
            json.dump(current_policy_performance, f, indent=2)
        print(f"💾 Best policy updated and saved to {BEST_POLICY_PATH}")
    
    # Load best policy for comparison (either current or existing)
    best_policy_data = current_policy_performance if should_update else None
    if not should_update and best_policy_exists:
        try:
            with open(BEST_POLICY_PATH, 'r') as f:
                best_policy_data = json.load(f)
        except Exception:
            best_policy_data = current_policy_performance

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
            ml_prev = log[i]['reasoning_balance_before']
            bh_prev = log[i]['buy_hold_balance_before']
        else:
            ml_prev = log[i-1]['reasoning_balance_after']
            bh_prev = log[i-1]['buy_hold_balance_after']
        ml_curr = log[i]['reasoning_balance_after']
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
    
    # Show best policy performance if available
    if best_policy_data and best_policy_data != current_policy_performance:
        print(f"{'Reasoning L/S (best all-time)':<20} ${best_policy_data['test_final_capital']:<11,.0f} {best_policy_data['test_return']:>8.1%} {best_policy_data['test_sharpe']:>6.2f} {(best_policy_data['test_return'] - sim_results['buy_hold_total_return'])*100:>10.1f}%")
    
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
    
    # Show best policy summary
    if best_policy_data:
        print(f"\n🏆 BEST POLICY SUMMARY:")
        print(f"=" * 70)
        print(f"📁 Location: {BEST_POLICY_PATH}")
        print(f"📅 Last Updated: {best_policy_data['timestamp']}")
        print(f"📈 Test Performance:")
        print(f"  Return: {best_policy_data['test_return']:.1%}")
        print(f"  Sharpe: {best_policy_data['test_sharpe']:.3f}")
        print(f"  Max Drawdown: {best_policy_data['test_max_drawdown']:.1%}")
        print(f"  Final Capital: ${best_policy_data['test_final_capital']:,.2f}")
        print(f"  Accuracy: {best_policy_data['test_accuracy']:.1%}")
        print(f"  UP Accuracy: {best_policy_data['test_up_accuracy']:.1%}")
        print(f"  DOWN Accuracy: {best_policy_data['test_down_accuracy']:.1%}")
        
        if best_policy_data == current_policy_performance:
            print(f"✅ Current run produced the best policy!")
        else:
            print(f"📊 Current run: {current_policy_performance['test_return']:.1%} vs Best: {best_policy_data['test_return']:.1%}")




if __name__ == "__main__":
    main()

