#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import time

def simplified_model_test():
    """Simplified ML model testing focused on training and evaluation metrics"""
    
    print("🚀 MLSolana Simplified Model Test")
    print("=" * 60)
    
    # Step 1: Load the feature matrix
    print("📊 Loading feature matrix...")
    df = pd.read_csv('feature_matrix.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: Prepare features and target
    print("\n🔧 Preparing data...")
    
    # Convert date to datetime and sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Select feature columns (exclude raw prices, future-looking data, and target)
    # Also exclude target_next_day_rolling_mean_2d as it uses future target information
    exclude_cols = ['date', 'sol_close', 'sol_actual_next_day_return', 'btc_close', 'eth_close', 'target_next_day', 'target_next_day_rolling_mean_2d']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Features to use for training
    X = df[feature_cols].copy()
    y = df['target_next_day'].copy()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Using features: {feature_cols}")
    print(f"✅ Including all returns, technical indicators, and macro features (FEDFUNDS, DXY, VIX)")
    
    # Handle missing values
    missing_counts = X.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"⚠️  Missing values found: {total_missing}")
        X = X.ffill().bfill()
        print("✅ Missing values filled")
    else:
        print("✅ No missing values found")
    
    # Check target distribution
    print(f"\n📈 Target distribution:")
    target_counts = y.value_counts().sort_index()
    target_labels = {-1: "Down (<0%)", 1: "Up (>=0%)"}
    
    for target, count in target_counts.items():
        percentage = (count / len(y)) * 100
        label = target_labels.get(target, f"Unknown ({target})")
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Step 3: Chronological train-test split (80% train, 20% test)
    print(f"\n🔄 Creating CHRONOLOGICAL train/test split (80% train, 20% test)...")
    
    # Split chronologically - first 80% for training, last 20% for testing
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    print(f"Training period: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} days)")
    print(f"Testing period:  {test_data['date'].min().date()} to {test_data['date'].max().date()} ({len(test_data)} days)")
    
    # Prepare train/test sets
    X_train = train_data[feature_cols].copy()
    y_train = train_data['target_next_day'].copy()
    X_test = test_data[feature_cols].copy()
    y_test = test_data['target_next_day'].copy()
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Define models with hyperparameter grids
    print(f"\n🤖 Setting up models...")
    
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 15, None],
                'model__min_samples_split': [5, 10],
                'model__min_samples_leaf': [2, 4]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [6, 8, 10],
                'model__learning_rate': [0.1, 0.2],
                'model__subsample': [0.8, 1.0]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model__n_estimators': [100, 150],
                'model__max_depth': [5, 7],
                'model__learning_rate': [0.1, 0.15],
                'model__subsample': [0.8, 1.0]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'model__hidden_layer_sizes': [(100,), (100, 50)],
                'model__alpha': [0.0001, 0.001],
                'model__learning_rate_init': [0.001, 0.01]
            }
        }
    }
    
    # Step 5: Train and evaluate models
    print(f"\n🔍 Training and optimizing models...\n")
    
    results = {}
    best_models = {}
    
    # Prepare cross-validation - USE TIME SERIES SPLIT TO PREVENT DATA LEAKAGE
    print("🚨 Using TimeSeriesSplit for proper time series cross-validation (NO DATA LEAKAGE)")
    cv = TimeSeriesSplit(n_splits=5)  # Respects temporal order
    
    for name, model_info in models.items():
        print(f"🔧 Optimizing {name}...")
        start_time = time.time()
        
        # Create pipeline with scaling for algorithms that need it
        if name in ['SVM', 'Neural Network']:
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', model_info['model'])
            ])
        else:
            pipeline = Pipeline([
                ('model', model_info['model'])
            ])
        
        # Handle XGBoost label mapping (needs 0,1 instead of -1,1)
        if name == 'XGBoost':
            # Map labels for XGBoost
            y_train_mapped = y_train.map({-1: 0, 1: 1})
            y_test_mapped = y_test.map({-1: 0, 1: 1})
            
            # Grid search
            grid_search = GridSearchCV(
                pipeline, 
                model_info['params'], 
                cv=cv, 
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train_mapped)
            
            # Get predictions and map back
            y_pred_mapped = grid_search.predict(X_test)
            y_pred = pd.Series(y_pred_mapped).map({0: -1, 1: 1}).values
            
            # Calculate metrics using original labels
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            cv_score = grid_search.best_score_
            
        else:
            # Regular training for other models
            grid_search = GridSearchCV(
                pipeline, 
                model_info['params'], 
                cv=cv, 
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Predictions
            y_pred = grid_search.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            cv_score = grid_search.best_score_
        
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'cv_score': cv_score,
            'training_time': training_time,
            'predictions': y_pred,
            'best_params': grid_search.best_params_
        }
        
        best_models[name] = grid_search.best_estimator_
        
        print(f"   ✅ Best CV F1: {cv_score:.3f}")
        print(f"   ⏱️  Training time: {training_time:.1f}s")
    
    # Step 6: Create ensemble model
    print(f"\n🎭 Creating ensemble model...")
    
    # Get top 3 models for ensemble
    sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_score'], reverse=True)
    top_3_models = [(name, best_models[name]) for name, _ in sorted_models[:3]]
    
    ensemble = VotingClassifier(
        estimators=top_3_models,
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    
    # Ensemble metrics
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1_weighted = f1_score(y_test, ensemble_pred, average='weighted')
    ensemble_f1_macro = f1_score(y_test, ensemble_pred, average='macro')
    
    results['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'f1_weighted': ensemble_f1_weighted,
        'f1_macro': ensemble_f1_macro,
        'cv_score': 'N/A',
        'predictions': ensemble_pred
    }
    
    best_models['Ensemble'] = ensemble
    
    # Step 7: Display comprehensive results
    print(f"\n📊 COMPREHENSIVE MODEL RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Weighted':<12} {'F1-Macro':<10} {'CV Score':<10}")
    print("-" * 70)
    
    # Sort by accuracy for display
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, metrics in sorted_results:
        cv_display = f"{metrics['cv_score']:.3f}" if metrics['cv_score'] != 'N/A' else 'N/A'
        print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['f1_weighted']:<12.3f} {metrics['f1_macro']:<10.3f} {cv_display:<10}")
    
    # Step 8: Detailed analysis of best model
    best_model_name = sorted_results[0][0]
    best_metrics = sorted_results[0][1]
    best_predictions = best_metrics['predictions']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print("=" * 50)
    
    print(f"\n📈 Detailed Performance:")
    
    # Classification report
    report = classification_report(y_test, best_predictions, 
                                 target_names=['Down', 'Up'], 
                                 output_dict=True)
    
    for class_name, metrics in report.items():
        if class_name in ['Down', 'Up']:
            print(f"  {class_name:<8}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_predictions)
    print(f"\n🔄 Confusion Matrix:")
    print(f"       Predicted")
    print(f"     Down   Up")
    print(f"     ────  ────")
    for i, actual in enumerate(['Down', 'Up']):
        print(f"{actual:<4} {cm[i][0]:4d}  {cm[i][1]:4d}")
    
    # Binary accuracy (same as overall accuracy now)
    binary_accuracy = accuracy_score(y_test, best_predictions)
    print(f"\n🎯 Binary Classification Accuracy: {binary_accuracy:.3f} ({binary_accuracy:.1%})")
    
    # Class-specific accuracy
    print(f"\n📊 Class-specific Performance:")
    for class_val, class_name in [(-1, 'Down'), (1, 'Up')]:
        class_mask = (y_test == class_val)
        if class_mask.sum() > 0:
            class_accuracy = (best_predictions[class_mask] == class_val).mean()
            print(f"  When actual was {class_name:<4}: {class_accuracy:.1%} predicted correctly")
    
    print(f"\n🎉 Model evaluation completed!")
    print(f"📊 Best model: {best_model_name} with {best_metrics['accuracy']:.1%} accuracy")
    
    # Step 9: Simple Trading Simulation
    print(f"\n" + "=" * 70)
    print(f"💰 SIMPLE TRADING SIMULATION")
    print(f"=" * 70)
    
    # Get test data for simulation
    test_dates = test_data['date'].values
    test_actual_returns = test_data['sol_actual_next_day_return'].values / 100  # Convert % to decimal
    test_actual_targets = test_data['target_next_day'].values  # Actual -1/1 classifications
    best_predictions = best_metrics['predictions']
    
    # Transaction cost parameters
    transaction_cost_per_trade = 0.002  # 0.2% per round trip (0.1% buy + 0.1% sell)
    
    print(f"📊 Simulation Setup:")
    print(f"  Period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    print(f"  Trading days: {len(test_dates)}")
    print(f"  Initial capital: $10,000")
    print(f"  Strategy: Long when predict UP (+1), Short when predict DOWN (-1)")
    print(f"  Transaction costs: {transaction_cost_per_trade*100:.1f}% per round trip (0.1% buy + 0.1% sell)")
    print()
    
    # Initialize simulation
    initial_capital = 10000
    ml_capital = initial_capital
    buy_hold_capital = initial_capital
    
    # Apply initial transaction cost for buy & hold strategy (0.1% to buy)
    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)
    
    # Track detailed results for CSV
    detailed_log = []
    
    print(f"📈 Day-by-Day Trading Results:")
    print(f"{'Date':<12} {'Pred':<4} {'Actual%':<8} {'Net Strat%':<10} {'ML Value':<10} {'B&H Value':<10}")
    print(f"-" * 70)
    
    for i, (date, prediction, actual_return, actual_target) in enumerate(zip(test_dates, best_predictions, test_actual_returns, test_actual_targets)):
        # Skip if no actual return data
        if pd.isna(actual_return) or pd.isna(actual_target):
            continue
            
        # Calculate strategy return
        if prediction == 1:  # Predicted UP - go long
            strategy_return = actual_return  # Gain/lose with market
        else:  # Predicted DOWN - go short  
            strategy_return = -actual_return  # Gain when market down, lose when market up
        
        # Apply transaction costs to ML strategy (0.2% per day for round trip)
        strategy_return_after_costs = strategy_return - transaction_cost_per_trade
        
        # Update portfolios
        ml_capital_before = ml_capital
        buy_hold_capital_before = buy_hold_capital
        
        ml_capital = ml_capital * (1 + strategy_return_after_costs)
        buy_hold_capital = buy_hold_capital * (1 + actual_return)
        
        # Check if prediction is correct
        prediction_correct = (prediction == actual_target)
        
        # Log detailed data for CSV (will add long-only data later)
        detailed_log.append({
            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
            'actual_target': int(actual_target),  # -1 or 1
            'prediction': int(prediction),  # -1 or 1
            'prediction_correct': prediction_correct,
            'actual_return_pct': actual_return * 100,
            'strategy_return_pct': strategy_return * 100,
            'strategy_return_after_costs_pct': strategy_return_after_costs * 100,
            'ml_balance_before': ml_capital_before,
            'ml_balance_after': ml_capital,
            'buy_hold_balance_before': buy_hold_capital_before,
            'buy_hold_balance_after': buy_hold_capital,
            # Long-only placeholders (will be filled in next step)
            'long_only_position': None,
            'long_only_prediction_correct': None,
            'long_only_strategy_return_pct': None,
            'long_only_balance_before': None,
            'long_only_balance_after': None
        })
        
        # Print every 10th day + first/last few days for display
        if i < 5 or i >= len(test_dates) - 5 or i % 10 == 0:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            pred_str = "UP" if prediction == 1 else "DOWN"
            actual_return_pct = actual_return * 100
            strategy_return_after_costs_pct = strategy_return_after_costs * 100
            print(f"{date_str:<12} {pred_str:<4} {actual_return_pct:>7.1f}% {strategy_return_after_costs_pct:>9.1f}% ${ml_capital:>9.0f} ${buy_hold_capital:>10.0f}")
        elif i == 5:
            print(f"{'...':<12} {'...':<4} {'...':<8} {'...':<10} {'...':<10} {'...':<10}")
    
    # Apply final transaction cost for buy & hold strategy (0.1% to sell at the end)
    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)
    
    # Update the last entry in detailed_log with final buy & hold balance
    if detailed_log:
        detailed_log[-1]['buy_hold_balance_after'] = buy_hold_capital
    
    # Save detailed CSV (will be updated with long-only data later)
    detailed_df = pd.DataFrame(detailed_log)
    detailed_df.to_csv('daily_trading_results.csv', index=False)
    print(f"\n✅ Detailed daily results saved to: daily_trading_results.csv (long-only data will be added)")
    
    # Calculate daily returns for Sharpe ratio analysis (consistent method for all days)
    ml_daily_returns = []
    buy_hold_daily_returns = []
    
    for i in range(len(detailed_log)):
        # Use consistent calculation: (end_balance - prev_end_balance) / prev_end_balance
        if i == 0:
            # First day: use starting balance as previous balance
            ml_prev_balance = detailed_log[i]['ml_balance_before']
            buy_hold_prev_balance = detailed_log[i]['buy_hold_balance_before']
        else:
            # Subsequent days: use previous day's ending balance
            ml_prev_balance = detailed_log[i-1]['ml_balance_after']
            buy_hold_prev_balance = detailed_log[i-1]['buy_hold_balance_after']
            
        ml_curr_balance = detailed_log[i]['ml_balance_after']
        buy_hold_curr_balance = detailed_log[i]['buy_hold_balance_after']
        
        ml_return = (ml_curr_balance - ml_prev_balance) / ml_prev_balance
        buy_hold_return = (buy_hold_curr_balance - buy_hold_prev_balance) / buy_hold_prev_balance
        
        ml_daily_returns.append(ml_return)
        buy_hold_daily_returns.append(buy_hold_return)
    
    # Calculate Sharpe ratios (assuming 0% risk-free rate for simplicity)
    # Note: For risk-adjusted returns with risk-free rate, use:
    # risk_free_daily = 0.05 / 252  # e.g., 5% annualized risk-free rate
    # excess_returns = [r - risk_free_daily for r in daily_returns]
    
    ml_mean_return = np.mean(ml_daily_returns)
    ml_std_return = np.std(ml_daily_returns, ddof=1)
    ml_sharpe_ratio = ml_mean_return / ml_std_return if ml_std_return > 0 else 0
    
    buy_hold_mean_return = np.mean(buy_hold_daily_returns)
    buy_hold_std_return = np.std(buy_hold_daily_returns, ddof=1)
    buy_hold_sharpe_ratio = buy_hold_mean_return / buy_hold_std_return if buy_hold_std_return > 0 else 0
    
    # Annualize Sharpe ratios (multiply by sqrt(252) for daily data)
    ml_sharpe_annual = ml_sharpe_ratio * np.sqrt(252)
    buy_hold_sharpe_annual = buy_hold_sharpe_ratio * np.sqrt(252)
    
    # Calculate final metrics
    ml_total_return = (ml_capital - initial_capital) / initial_capital
    buy_hold_total_return = (buy_hold_capital - initial_capital) / initial_capital
    
    # Calculate metrics from detailed log
    total_trading_days = len(detailed_log)
    winning_days = sum(1 for day in detailed_log if day['prediction_correct'])
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
    
    print()
    print(f"📊 FINAL SIMULATION RESULTS:")
    print(f"=" * 50)
    print(f"🤖 ML Trading Strategy:")
    print(f"  Final Portfolio Value: ${ml_capital:,.2f}")
    print(f"  Total Return: {ml_total_return:.1%}")
    print(f"  Win Rate: {win_rate:.1%} ({winning_days}/{total_trading_days} days)")
    print(f"  Daily Mean Return: {ml_mean_return:.4f} ({ml_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {ml_std_return:.4f} ({ml_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio (Annualized): {ml_sharpe_annual:.3f}")
    print()
    print(f"📈 Buy & Hold Strategy:")
    print(f"  Final Portfolio Value: ${buy_hold_capital:,.2f}")
    print(f"  Total Return: {buy_hold_total_return:.1%}")
    print(f"  Daily Mean Return: {buy_hold_mean_return:.4f} ({buy_hold_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {buy_hold_std_return:.4f} ({buy_hold_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio (Annualized): {buy_hold_sharpe_annual:.3f}")
    print()
    
    # Compare strategies
    excess_return = ml_total_return - buy_hold_total_return
    if excess_return > 0:
        print(f"🎯 ML Strategy BEATS Buy & Hold by {excess_return:.1%}")
        outperformance = ((ml_capital / buy_hold_capital) - 1) * 100
        print(f"🏆 Outperformance: {outperformance:.1f}%")
    else:
        print(f"📉 ML Strategy UNDERPERFORMS Buy & Hold by {abs(excess_return):.1%}")
        underperformance = ((buy_hold_capital / ml_capital) - 1) * 100
        print(f"📊 Underperformance: {underperformance:.1f}%")
    
    # Sharpe ratio comparison
    print(f"\n📊 RISK-ADJUSTED PERFORMANCE (Sharpe Ratio):")
    sharpe_diff = ml_sharpe_annual - buy_hold_sharpe_annual
    if sharpe_diff > 0:
        print(f"🎯 ML Strategy has BETTER risk-adjusted returns: {ml_sharpe_annual:.3f} vs {buy_hold_sharpe_annual:.3f}")
        print(f"🏆 Sharpe Ratio Improvement: +{sharpe_diff:.3f}")
    else:
        print(f"📉 ML Strategy has WORSE risk-adjusted returns: {ml_sharpe_annual:.3f} vs {buy_hold_sharpe_annual:.3f}")
        print(f"🔻 Sharpe Ratio Decline: {abs(sharpe_diff):.3f}")
    
    # Sharpe ratio interpretation
    print(f"\n💡 Sharpe Ratio Interpretation:")
    if ml_sharpe_annual > 1.0:
        print(f"  🔥 ML Strategy Sharpe > 1.0: EXCELLENT risk-adjusted performance")
    elif ml_sharpe_annual > 0.5:
        print(f"  ✅ ML Strategy Sharpe > 0.5: GOOD risk-adjusted performance")
    elif ml_sharpe_annual > 0.0:
        print(f"  ⚠️  ML Strategy Sharpe > 0.0: POSITIVE but modest risk-adjusted performance")
    else:
        print(f"  ❌ ML Strategy Sharpe < 0.0: NEGATIVE risk-adjusted performance")
    
    print(f"\n💡 Strategy Analysis:")
    
    # Calculate UP signal accuracy
    up_predictions = [day for day in detailed_log if day['prediction'] == 1]
    correct_up_predictions = [day for day in up_predictions if day['prediction_correct']]
    
    # Calculate DOWN signal accuracy  
    down_predictions = [day for day in detailed_log if day['prediction'] == -1]
    correct_down_predictions = [day for day in down_predictions if day['prediction_correct']]
    
    if len(up_predictions) > 0:
        up_accuracy = len(correct_up_predictions) / len(up_predictions)
        print(f"  UP signal accuracy: {up_accuracy:.1%} ({len(correct_up_predictions)}/{len(up_predictions)})")
    if len(down_predictions) > 0:
        down_accuracy = len(correct_down_predictions) / len(down_predictions)  
        print(f"  DOWN signal accuracy: {down_accuracy:.1%} ({len(correct_down_predictions)}/{len(down_predictions)})")
    
    print(f"  CSV contains {len(detailed_log)} trading days with full details")
    
    print(f"\n🎉 Trading simulation completed!")
    
    # Step 10: Long-Only Strategy Variant
    print(f"\n" + "=" * 70)
    print(f"📈 LONG-ONLY STRATEGY SIMULATION")
    print(f"=" * 70)
    
    print(f"📊 Long-Only Setup:")
    print(f"  Strategy: Long when predict UP (+1), Cash when predict DOWN (-1)")
    print(f"  No shorting - cash earns 0% during DOWN predictions")
    print(f"  Period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    print()
    
    # Initialize long-only simulation
    long_only_capital = initial_capital
    
    print(f"📈 Long-Only Day-by-Day Results:")
    print(f"{'Date':<12} {'Pred':<4} {'Position':<8} {'Actual%':<8} {'Net Strat%':<10} {'Long Value':<10}")
    print(f"-" * 75)
    
    # Update existing detailed_log with long-only data
    for i, (date, prediction, actual_return, actual_target) in enumerate(zip(test_dates, best_predictions, test_actual_returns, test_actual_targets)):
        # Skip if no actual return data
        if pd.isna(actual_return) or pd.isna(actual_target):
            continue
            
        # Long-only strategy logic
        if prediction == 1:  # Predicted UP - go long
            position = "LONG"
            strategy_return = actual_return  # Gain/lose with market
            # Apply transaction costs for long position (0.2% round trip)
            strategy_return_after_costs = strategy_return - transaction_cost_per_trade
        else:  # Predicted DOWN - stay in cash
            position = "CASH"
            strategy_return = 0.0  # Cash earns nothing
            strategy_return_after_costs = 0.0  # No transaction cost for cash
        
        # Update long-only portfolio
        long_only_capital_before = long_only_capital
        long_only_capital = long_only_capital * (1 + strategy_return_after_costs)
        
        # Check if prediction is correct for long-only context
        if prediction == 1:
            # For long predictions, correct if market went up
            long_only_prediction_correct = (actual_return > 0)
        else:
            # For cash predictions, correct if market went down (avoided loss)
            long_only_prediction_correct = (actual_return < 0)
        
        # Find corresponding entry in detailed_log and update with long-only data
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        for log_entry in detailed_log:
            if log_entry['date'] == date_str:
                log_entry['long_only_position'] = position
                log_entry['long_only_prediction_correct'] = long_only_prediction_correct
                log_entry['long_only_strategy_return_pct'] = strategy_return_after_costs * 100
                log_entry['long_only_balance_before'] = long_only_capital_before
                log_entry['long_only_balance_after'] = long_only_capital
                break
        
        # Print every 10th day + first/last few days for display
        if i < 5 or i >= len(test_dates) - 5 or i % 10 == 0:
            pred_str = "UP" if prediction == 1 else "DOWN"
            actual_return_pct = actual_return * 100
            strategy_return_after_costs_pct = strategy_return_after_costs * 100
            print(f"{date_str:<12} {pred_str:<4} {position:<8} {actual_return_pct:>7.1f}% {strategy_return_after_costs_pct:>9.1f}% ${long_only_capital:>9.0f}")
        elif i == 5:
            print(f"{'...':<12} {'...':<4} {'...':<8} {'...':<8} {'...':<10} {'...':<10}")
    
    # Calculate long-only daily returns for Sharpe ratio (consistent method)
    valid_entries = [day for day in detailed_log if day['long_only_position'] is not None]
    long_only_daily_returns = []
    
    for i in range(len(valid_entries)):
        # Use consistent calculation: (end_balance - prev_end_balance) / prev_end_balance
        if i == 0:
            # First day: use starting balance as previous balance
            prev_balance = valid_entries[i]['long_only_balance_before']
        else:
            # Subsequent days: use previous day's ending balance
            prev_balance = valid_entries[i-1]['long_only_balance_after']
            
        curr_balance = valid_entries[i]['long_only_balance_after']
        ret = (curr_balance - prev_balance) / prev_balance
        long_only_daily_returns.append(ret)
    
    # Calculate long-only Sharpe ratio
    long_only_mean_return = np.mean(long_only_daily_returns)
    long_only_std_return = np.std(long_only_daily_returns, ddof=1)
    long_only_sharpe_ratio = long_only_mean_return / long_only_std_return if long_only_std_return > 0 else 0
    long_only_sharpe_annual = long_only_sharpe_ratio * np.sqrt(252)
    
    # Calculate long-only final metrics
    long_only_total_return = (long_only_capital - initial_capital) / initial_capital
    
    # Calculate long-only metrics using updated detailed_log
    total_long_only_days = len(valid_entries)
    long_only_winning_days = sum(1 for day in valid_entries if day['long_only_prediction_correct'])
    long_only_win_rate = long_only_winning_days / total_long_only_days if total_long_only_days > 0 else 0
    
    # Calculate position breakdown
    long_positions = [day for day in valid_entries if day['long_only_position'] == 'LONG']
    cash_positions = [day for day in valid_entries if day['long_only_position'] == 'CASH']
    
    print()
    print(f"📊 LONG-ONLY FINAL RESULTS:")
    print(f"=" * 50)
    print(f"📈 Long-Only Strategy:")
    print(f"  Final Portfolio Value: ${long_only_capital:,.2f}")
    print(f"  Total Return: {long_only_total_return:.1%}")
    print(f"  Win Rate: {long_only_win_rate:.1%} ({long_only_winning_days}/{total_long_only_days} days)")
    print(f"  Daily Mean Return: {long_only_mean_return:.4f} ({long_only_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {long_only_std_return:.4f} ({long_only_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio (Annualized): {long_only_sharpe_annual:.3f}")
    print(f"  Long Positions: {len(long_positions)} days ({len(long_positions)/total_long_only_days*100:.1f}%)")
    print(f"  Cash Positions: {len(cash_positions)} days ({len(cash_positions)/total_long_only_days*100:.1f}%)")
    print()
    
    # Compare all three strategies
    print(f"🏆 STRATEGY COMPARISON:")
    print(f"=" * 70)
    print(f"{'Strategy':<15} {'Final Value':<12} {'Return':<10} {'Sharpe':<8} {'vs Buy&Hold':<12}")
    print(f"-" * 70)
    print(f"{'ML Long/Short':<15} ${ml_capital:<11,.0f} {ml_total_return:>8.1%} {ml_sharpe_annual:>6.2f} {(ml_total_return - buy_hold_total_return)*100:>10.1f}%")
    print(f"{'Long-Only':<15} ${long_only_capital:<11,.0f} {long_only_total_return:>8.1%} {long_only_sharpe_annual:>6.2f} {(long_only_total_return - buy_hold_total_return)*100:>10.1f}%")
    print(f"{'Buy & Hold':<15} ${buy_hold_capital:<11,.0f} {buy_hold_total_return:>8.1%} {buy_hold_sharpe_annual:>6.2f} {'0.0%':>10}")
    
    # Long-only vs ML strategy comparison
    long_only_vs_ml = long_only_total_return - ml_total_return
    if long_only_vs_ml > 0:
        print(f"\n🎯 Long-Only BEATS ML Strategy by {long_only_vs_ml:.1%}")
    else:
        print(f"\n📉 Long-Only UNDERPERFORMS ML Strategy by {abs(long_only_vs_ml):.1%}")
    
    print(f"\n💡 Long-Only Strategy Analysis:")
    
    # Long position accuracy
    correct_long_positions = [day for day in long_positions if day['long_only_prediction_correct']]
    if len(long_positions) > 0:
        long_accuracy = len(correct_long_positions) / len(long_positions)
        print(f"  LONG position accuracy: {long_accuracy:.1%} ({len(correct_long_positions)}/{len(long_positions)})")
    
    # Cash position accuracy (avoided losses)
    correct_cash_positions = [day for day in cash_positions if day['long_only_prediction_correct']]
    if len(cash_positions) > 0:
        cash_accuracy = len(correct_cash_positions) / len(cash_positions)
        print(f"  CASH position accuracy: {cash_accuracy:.1%} ({len(correct_cash_positions)}/{len(cash_positions)}) - avoided losses")
    
    print(f"  Combined CSV contains {len(valid_entries)} trading days with both long/short and long-only details")
    
    # Update CSV with complete data (both strategies)
    updated_df = pd.DataFrame(detailed_log)
    updated_df.to_csv('daily_trading_results.csv', index=False)
    print(f"\n✅ Updated daily_trading_results.csv with both Long/Short AND Long-Only strategy data")
    
    print(f"\n🎉 Long-only simulation completed!")
    
    return results, best_models

if __name__ == "__main__":
    results, models = simplified_model_test() 