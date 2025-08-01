#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
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
    
    # Step 3: Chronological train-test split (70% train, 30% test)
    print(f"\n🔄 Creating CHRONOLOGICAL train/test split (70% train, 30% test)...")
    
    # Split chronologically - first 70% for training, last 30% for testing
    split_idx = int(len(df) * 0.7)
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
        if name in ['SVM']:
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
    
    # Feature Importance Analysis
    print(f"\n" + "=" * 70)
    print(f"🎯 FEATURE IMPORTANCE ANALYSIS")
    print(f"=" * 70)
    
    best_model_obj = best_models[best_model_name]
    
    # Extract feature importance based on model type
    if hasattr(best_model_obj, 'feature_importances_'):
        # Tree-based models (RF, XGBoost, GB)
        feature_importance = best_model_obj.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(best_model_obj, 'coef_'):
        # Linear models (SVM with linear kernel)
        feature_importance = np.abs(best_model_obj.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        # Try to get from pipeline
        if hasattr(best_model_obj.named_steps['model'], 'feature_importances_'):
            feature_importance = best_model_obj.named_steps['model'].feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(best_model_obj.named_steps['model'], 'coef_'):
            feature_importance = np.abs(best_model_obj.named_steps['model'].coef_[0])
            importance_type = "Coefficient Magnitude"
        else:
            feature_importance = None
    
    if feature_importance is not None:
        # Create feature importance dataframe
        feature_names = [col for col in df.columns if col not in exclude_cols]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"📊 Top 15 Most Important Features ({importance_type}):")
        print(f"-" * 50)
        for i, row in importance_df.head(15).iterrows():
            print(f"{row['Feature']:<35} {row['Importance']:.4f}")
            
        print(f"\n💡 Key Insights:")
        top_3 = importance_df.head(3)['Feature'].tolist()
        print(f"  🥇 Top 3 features: {', '.join(top_3)}")
        
        # Analyze feature categories
        returns_features = [f for f in feature_names if 'return' in f]
        tech_features = [f for f in feature_names if any(x in f for x in ['rsi', 'macd', 'sma', 'volatility'])]
        macro_features = [f for f in feature_names if f in ['fedfunds', 'dxy', 'vix']]
        relative_features = [f for f in feature_names if 'relative' in f]
        
        returns_importance = importance_df[importance_df['Feature'].isin(returns_features)]['Importance'].sum()
        tech_importance = importance_df[importance_df['Feature'].isin(tech_features)]['Importance'].sum()
        macro_importance = importance_df[importance_df['Feature'].isin(macro_features)]['Importance'].sum()
        relative_importance = importance_df[importance_df['Feature'].isin(relative_features)]['Importance'].sum()
        
        print(f"  📊 Returns features total importance: {returns_importance:.3f}")
        print(f"  🔧 Technical indicators total importance: {tech_importance:.3f}")
        print(f"  🏛️ Macro features total importance: {macro_importance:.3f}")
        print(f"  🔗 Relative pricing total importance: {relative_importance:.3f}")
    else:
        print("⚠️  Feature importance not available for this model type")
    
    # UP Bias Analysis
    print(f"\n" + "=" * 70)
    print(f"🎯 UP BIAS ANALYSIS")
    print(f"=" * 70)
    
    # Analyze test period distribution
    test_actual_up = (test_data['target_next_day'] == 1).sum()
    test_actual_down = (test_data['target_next_day'] == -1).sum()
    test_total = len(test_data)
    
    # Analyze model predictions
    test_pred_up = (best_predictions == 1).sum()
    test_pred_down = (best_predictions == -1).sum()
    
    # Calculate performance on each class
    test_targets = test_data['target_next_day'].values
    up_mask = test_targets == 1
    down_mask = test_targets == -1
    
    up_accuracy = accuracy_score(test_targets[up_mask], best_predictions[up_mask])
    down_accuracy = accuracy_score(test_targets[down_mask], best_predictions[down_mask])
    
    print(f"📊 Test Period Market Distribution:")
    print(f"  Actual UP days: {test_actual_up} ({test_actual_up/test_total*100:.1f}%)")
    print(f"  Actual DOWN days: {test_actual_down} ({test_actual_down/test_total*100:.1f}%)")
    
    print(f"\n🤖 Model Prediction Distribution:")
    print(f"  Predicted UP: {test_pred_up} ({test_pred_up/test_total*100:.1f}%)")
    print(f"  Predicted DOWN: {test_pred_down} ({test_pred_down/test_total*100:.1f}%)")
    
    print(f"\n📈 Class-Specific Accuracy:")
    print(f"  UP accuracy: {up_accuracy:.1%} ({(best_predictions[up_mask] == 1).sum()}/{up_mask.sum()} correct)")
    print(f"  DOWN accuracy: {down_accuracy:.1%} ({(best_predictions[down_mask] == -1).sum()}/{down_mask.sum()} correct)")
    
    # Analysis conclusion
    market_up_bias = test_actual_up / test_total
    model_up_bias = test_pred_up / test_total
    
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
    
    print(f"🔄 Running trading simulation...")
    
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
    
    print(f"\n🎉 Trading simulation completed!")
    
    # Step 10: Long-Only Strategy Variant
    print(f"\n" + "=" * 70)
    print(f"📈 LONG-ONLY STRATEGY SIMULATION")
    print(f"=" * 70)
    
    print(f"🔄 Calculating long-only strategy...")
    
    # Initialize long-only simulation
    long_only_capital = initial_capital
    
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
        
        # No day-by-day printing - results go to CSV
    
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
    # Update CSV with complete data (both strategies)
    updated_df = pd.DataFrame(detailed_log)
    updated_df.to_csv('daily_trading_results.csv', index=False)
    
    # Consolidated Simulation Results
    print(f"\n📊 COMPREHENSIVE SIMULATION RESULTS:")
    print(f"=" * 70)
    
    print(f"🤖 ML Long/Short Strategy:")
    print(f"  Final Portfolio Value: ${ml_capital:,.2f}")
    print(f"  Total Return: {ml_total_return:.1%}")
    print(f"  Daily Mean Return: {ml_mean_return:.4f} ({ml_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {ml_std_return:.4f} ({ml_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio: {ml_sharpe_annual:.3f}")
    print(f"  Win Rate: {win_rate:.1%} ({winning_days}/{total_trading_days} days)")
    
    print(f"\n📈 Long-Only Strategy:")
    print(f"  Final Portfolio Value: ${long_only_capital:,.2f}")
    print(f"  Total Return: {long_only_total_return:.1%}")
    print(f"  Daily Mean Return: {long_only_mean_return:.4f} ({long_only_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {long_only_std_return:.4f} ({long_only_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio: {long_only_sharpe_annual:.3f}")
    print(f"  Long Positions: {len(long_positions)} days ({len(long_positions)/total_long_only_days*100:.1f}%) | Cash: {len(cash_positions)} days")
    
    print(f"\n📊 Buy & Hold Strategy:")
    print(f"  Final Portfolio Value: ${buy_hold_capital:,.2f}")
    print(f"  Total Return: {buy_hold_total_return:.1%}")
    print(f"  Daily Mean Return: {buy_hold_mean_return:.4f} ({buy_hold_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {buy_hold_std_return:.4f} ({buy_hold_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio: {buy_hold_sharpe_annual:.3f}")
    
    # Strategy Comparison Table
    print(f"\n🏆 STRATEGY COMPARISON:")
    print(f"=" * 70)
    print(f"{'Strategy':<15} {'Final Value':<12} {'Return':<10} {'Sharpe':<8} {'vs Buy&Hold':<12}")
    print(f"-" * 70)
    print(f"{'ML Long/Short':<15} ${ml_capital:<11,.0f} {ml_total_return:>8.1%} {ml_sharpe_annual:>6.2f} {(ml_total_return - buy_hold_total_return)*100:>10.1f}%")
    print(f"{'Long-Only':<15} ${long_only_capital:<11,.0f} {long_only_total_return:>8.1%} {long_only_sharpe_annual:>6.2f} {(long_only_total_return - buy_hold_total_return)*100:>10.1f}%")
    print(f"{'Buy & Hold':<15} ${buy_hold_capital:<11,.0f} {buy_hold_total_return:>8.1%} {buy_hold_sharpe_annual:>6.2f} {'0.0%':>10}")
    
    # Risk-Adjusted Performance Summary
    print(f"\n📊 RISK-ADJUSTED PERFORMANCE:")
    sharpe_diff_ml = ml_sharpe_annual - buy_hold_sharpe_annual
    sharpe_diff_lo = long_only_sharpe_annual - buy_hold_sharpe_annual
    
    if ml_sharpe_annual > 1.0:
        ml_rating = "🔥 EXCELLENT"
    elif ml_sharpe_annual > 0.5:
        ml_rating = "✅ GOOD"
    else:
        ml_rating = "⚠️ MODERATE"
        
    if long_only_sharpe_annual > 1.0:
        lo_rating = "🔥 EXCELLENT"
    elif long_only_sharpe_annual > 0.5:
        lo_rating = "✅ GOOD"
    else:
        lo_rating = "⚠️ MODERATE"
    
    print(f"  ML Strategy: {ml_rating} ({ml_sharpe_annual:.3f} vs {buy_hold_sharpe_annual:.3f}, +{sharpe_diff_ml:.3f})")
    print(f"  Long-Only: {lo_rating} ({long_only_sharpe_annual:.3f} vs {buy_hold_sharpe_annual:.3f}, +{sharpe_diff_lo:.3f})")
    
    # Transaction Cost Impact
    total_trades = len([d for d in detailed_log if d['prediction'] != 0])
    total_cost_pct = total_trades * transaction_cost_per_trade * 100
    print(f"\n💸 TRANSACTION COST IMPACT:")
    print(f"  Total trades: {total_trades} | Cost per trade: {transaction_cost_per_trade*100:.1f}% | Total cost: {total_cost_pct:.1f}%")
    
    print(f"\n✅ Complete trading data saved to: daily_trading_results.csv")
    print(f"🎉 Simulation completed - {len(detailed_log)} trading days analyzed")
    
    return results, best_models

if __name__ == "__main__":
    results, models = simplified_model_test() 