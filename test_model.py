#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
import time

def advanced_model_test():
    """Advanced ML model testing with hyperparameter optimization and multiple algorithms"""
    
    print("🚀 MLSolana Advanced Model Test")
    print("=" * 60)
    
    # Step 1: Load the feature matrix (READ-ONLY)
    print("📊 Loading feature matrix (read-only)...")
    df = pd.read_csv('feature_matrix.csv')
    
    # Create a deep copy to ensure we never modify the original data
    df_work = df.copy()
    
    print(f"Dataset shape: {df_work.shape}")
    print(f"Date range: {df_work['date'].min()} to {df_work['date'].max()}")
    
    # Step 2: Prepare features and target
    print("\n🔧 Preparing data...")
    
    # Select feature columns (exclude date, raw prices, future-looking data, and target columns)
    # Only use features starting from sol_return_1d onwards (no raw price levels or forward-looking data)
    exclude_cols = ['date', 'sol_close', 'sol_actual_next_day_return', 'btc_close', 'eth_close', 'target_next_day']
    feature_cols = [col for col in df_work.columns if col not in exclude_cols]
    
    X = df_work[feature_cols].copy()
    y = df_work['target_next_day'].copy()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Using features: {feature_cols}")
    print(f"Excluded raw prices: sol_close, btc_close, eth_close")
    
    # Handle missing values
    missing_counts = X.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"\n⚠️  Missing values found: {total_missing}")
        X = X.fillna(method='ffill').fillna(method='bfill')
        print("✅ Missing values filled")
    
    # Check target distribution
    print(f"\n📈 Target distribution:")
    target_counts = y.value_counts().sort_index()
    target_labels = {-1: "Down (<-2%)", 0: "Neutral (±2%)", 1: "Up (>+2%)"}
    
    for target, count in target_counts.items():
        percentage = (count / len(y)) * 100
        label = target_labels.get(target, f"Unknown ({target})")
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Step 3: Advanced train-test split with stratification
    print(f"\n🔄 Splitting data (80% train, 20% test)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Feature scaling (for algorithms that need it)
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Define multiple models with hyperparameter grids
    print(f"\n🤖 Setting up advanced models...")
    
    models = {}
    
    # 1. Random Forest with focused grid search
    rf_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    models['Random Forest'] = {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'param_grid': rf_param_grid,
        'use_scaled': False
    }
    
    # 2. XGBoost with reduced parameter grid (for faster training)
    xgb_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    models['XGBoost'] = {
        'model': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'param_grid': xgb_param_grid,
        'use_scaled': False,
        'needs_label_mapping': True  # XGBoost needs 0,1,2 instead of -1,0,1
    }
    
    # 3. Gradient Boosting
    gb_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'max_features': ['sqrt', 'log2']
    }
    
    models['Gradient Boosting'] = {
        'model': GradientBoostingClassifier(random_state=42),
        'param_grid': gb_param_grid,
        'use_scaled': False
    }
    
    # 4. SVM (works better with scaled features) - reduced grid
    svm_param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf']  # Focus on RBF kernel only
    }
    
    models['SVM'] = {
        'model': SVC(random_state=42, probability=True),
        'param_grid': svm_param_grid,
        'use_scaled': True
    }
    
    # 5. Neural Network - simplified grid
    mlp_param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu'],
        'alpha': [0.001, 0.01],
        'learning_rate': ['adaptive'],
        'max_iter': [500]
    }
    
    models['Neural Network'] = {
        'model': MLPClassifier(random_state=42, early_stopping=True),
        'param_grid': mlp_param_grid,
        'use_scaled': True
    }
    
    # Step 6: Train and optimize each model
    print(f"\n🔍 Training and optimizing models (this may take a while)...")
    
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    results = {}
    
    for name, model_config in models.items():
        print(f"\n🔧 Optimizing {name}...")
        start_time = time.time()
        
        # Choose appropriate data (scaled or not)
        X_train_use = X_train_scaled if model_config['use_scaled'] else X_train
        X_test_use = X_test_scaled if model_config['use_scaled'] else X_test
        
        # Handle label mapping for models that need it (like XGBoost)
        if model_config.get('needs_label_mapping', False):
            # Map our labels (-1, 0, 1) to (0, 1, 2) for XGBoost
            label_mapping = {-1: 0, 0: 1, 1: 2}
            reverse_mapping = {0: -1, 1: 0, 2: 1}
            y_train_mapped = y_train.map(label_mapping)
            y_test_mapped = y_test.map(label_mapping)
        else:
            y_train_mapped = y_train
            y_test_mapped = y_test
            reverse_mapping = None
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['param_grid'],
            cv=cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_use, y_train_mapped)
        
        # Get best model and make predictions
        best_model = grid_search.best_estimator_
        y_pred_mapped = best_model.predict(X_test_use)
        y_pred_proba = best_model.predict_proba(X_test_use)
        
        # Map predictions back to original labels if needed
        if reverse_mapping is not None:
            y_pred = pd.Series(y_pred_mapped).map(reverse_mapping).values
        else:
            y_pred = y_pred_mapped
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Store results
        best_models[name] = {
            'model': best_model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'use_scaled': model_config['use_scaled']
        }
        
        results[name] = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'training_time': time.time() - start_time
        }
        
        print(f"   ✅ Best CV F1: {grid_search.best_score_:.3f}")
        print(f"   ⏱️  Training time: {results[name]['training_time']:.1f}s")
    
    # Step 7: Create ensemble model
    print(f"\n🎭 Creating ensemble model...")
    
    # Use top 3 models for ensemble
    top_models = sorted(results.items(), key=lambda x: x[1]['f1_weighted'], reverse=True)[:3]
    
    ensemble_estimators = []
    for name, _ in top_models:
        model_info = best_models[name]
        ensemble_estimators.append((name.lower().replace(' ', '_'), model_info['model']))
    
    # Train ensemble
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    
    # Use appropriate training data for ensemble (mix of scaled and non-scaled)
    voting_clf.fit(X_train, y_train)  # Use non-scaled for ensemble
    ensemble_pred = voting_clf.predict(X_test)
    
    # Add ensemble results
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    
    results['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'f1_weighted': ensemble_f1,
        'f1_macro': f1_score(y_test, ensemble_pred, average='macro')
    }
    
    # Step 8: Display comprehensive results
    print(f"\n📊 COMPREHENSIVE MODEL RESULTS")
    print("=" * 70)
    
    # Sort models by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_weighted'], reverse=True)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Weighted':<12} {'F1-Macro':<10} {'CV Score':<10}")
    print("-" * 70)
    
    for name, metrics in sorted_results:
        cv_score = metrics.get('cv_score', 'N/A')
        cv_str = f"{cv_score:.3f}" if cv_score != 'N/A' else 'N/A'
        print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['f1_weighted']:<12.3f} "
              f"{metrics['f1_macro']:<10.3f} {cv_str:<10}")
    
    # Best model analysis
    best_model_name = sorted_results[0][0]
    best_model_info = best_models.get(best_model_name, {})
    
    if best_model_info:
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print("=" * 50)
        
        best_pred = best_model_info['predictions']
        
        # Detailed metrics for best model
        print(f"\n📈 Detailed Performance:")
        precision = precision_score(y_test, best_pred, average=None, labels=[-1, 0, 1])
        recall = recall_score(y_test, best_pred, average=None, labels=[-1, 0, 1])
        f1 = f1_score(y_test, best_pred, average=None, labels=[-1, 0, 1])
        
        class_names = ["Down", "Neutral", "Up"]
        for i, (name, p, r, f) in enumerate(zip(class_names, precision, recall, f1)):
            print(f"  {name:8}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
        
        # Confusion Matrix
        print(f"\n🔄 Confusion Matrix:")
        cm = confusion_matrix(y_test, best_pred, labels=[-1, 0, 1])
        
        print("       Predicted")
        print("     Down  Neut   Up")
        print("     ────  ────  ────")
        for i, (actual_label, row) in enumerate(zip(["Down", "Neut", "Up"], cm)):
            print(f"{actual_label:4} {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
        
        # Detailed TP/TN/FP/FN breakdown for each class
        print(f"\n📊 Detailed Classification Metrics:")
        class_names = ["Down (-1)", "Neutral (0)", "Up (+1)"]
        class_labels = [-1, 0, 1]
        
        for i, (class_name, class_label) in enumerate(zip(class_names, class_labels)):
            # Calculate TP, TN, FP, FN for this class
            tp = cm[i, i]  # True positives
            fn = np.sum(cm[i, :]) - tp  # False negatives (actual class, predicted other)
            fp = np.sum(cm[:, i]) - tp  # False positives (predicted class, actually other)
            tn = np.sum(cm) - tp - fn - fp  # True negatives
            
            print(f"\n  {class_name}:")
            print(f"    TP: {tp:3d} | TN: {tn:3d} | FP: {fp:3d} | FN: {fn:3d}")
            
            # Calculate metrics for this class
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"    Sensitivity (Recall): {sensitivity:.3f}")
            print(f"    Specificity:          {specificity:.3f}")
            print(f"    Precision:            {precision:.3f}")
        
        # Overall classification accuracy breakdown
        total_correct = np.trace(cm)
        total_samples = np.sum(cm)
        overall_accuracy = total_correct / total_samples
        
        print(f"\n🎯 Overall Classification Summary:")
        print(f"  Total Correct: {total_correct}/{total_samples} = {overall_accuracy:.3f}")
        print(f"  Total Wrong:   {total_samples - total_correct}/{total_samples} = {1 - overall_accuracy:.3f}")
        
        # Best parameters
        best_params = results[best_model_name].get('best_params', {})
        if best_params:
            print(f"\n⚙️  Best Parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
    
    # Feature importance (if available)
    if hasattr(best_model_info.get('model'), 'feature_importances_'):
        print(f"\n🔍 Feature Importance (Top 10):")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model_info['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (idx, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:25}: {row['importance']:.4f}")
    
    # Final insights
    print(f"\n💡 KEY INSIGHTS:")
    
    best_accuracy = sorted_results[0][1]['accuracy']
    random_accuracy = max(target_counts) / len(y)
    
    if best_accuracy > random_accuracy:
        improvement = ((best_accuracy - random_accuracy) / random_accuracy) * 100
        print(f"  ✅ Best model beats random by {improvement:.1f}%")
        print(f"     (Random: {random_accuracy:.3f}, Best: {best_accuracy:.3f})")
    else:
        print(f"  ⚠️  Models struggle to beat random guessing")
        print(f"     (Random: {random_accuracy:.3f}, Best: {best_accuracy:.3f})")
    
    # Enhanced Directional accuracy analysis
    if best_model_info:
        print(f"\n📈 DIRECTIONAL ACCURACY ANALYSIS:")
        
        # Up vs Down only (excluding neutral)
        mask_up_down = (y_test != 0) & (best_model_info['predictions'] != 0)
        if mask_up_down.sum() > 0:
            y_test_directional = y_test[mask_up_down]
            y_pred_directional = best_model_info['predictions'][mask_up_down]
            directional_acc = accuracy_score(y_test_directional, y_pred_directional)
            
            # Count directional predictions
            up_correct = np.sum((y_test_directional == 1) & (y_pred_directional == 1))
            down_correct = np.sum((y_test_directional == -1) & (y_pred_directional == -1))
            total_directional = len(y_test_directional)
            
            print(f"  🎯 Pure Directional Accuracy: {directional_acc:.3f} ({directional_acc*100:.1f}%)")
            print(f"  📊 Directional Breakdown:")
            print(f"    Correct UP predictions:   {up_correct}/{np.sum(y_test_directional == 1)}")
            print(f"    Correct DOWN predictions: {down_correct}/{np.sum(y_test_directional == -1)}")
            print(f"    Total directional samples: {total_directional}")
        
        # Include neutral in different analysis
        print(f"\n  🔄 All Predictions Breakdown:")
        for actual_class, actual_name in [(-1, "DOWN"), (0, "NEUTRAL"), (1, "UP")]:
            actual_mask = (y_test == actual_class)
            if actual_mask.sum() > 0:
                pred_for_actual = best_model_info['predictions'][actual_mask]
                correct_pct = np.mean(pred_for_actual == actual_class) * 100
                print(f"    When actual was {actual_name:7}: {correct_pct:5.1f}% predicted correctly")
        
        # Trading signal accuracy (if we only traded on strong signals)
        strong_up_mask = best_model_info['predictions'] == 1
        strong_down_mask = best_model_info['predictions'] == -1
        
        if strong_up_mask.sum() > 0:
            up_signal_accuracy = np.mean(y_test[strong_up_mask] == 1) * 100
            print(f"\n  📈 UP Signal Accuracy: {up_signal_accuracy:.1f}% ({np.sum(strong_up_mask)} signals)")
        
        if strong_down_mask.sum() > 0:
            down_signal_accuracy = np.mean(y_test[strong_down_mask] == -1) * 100
            print(f"  📉 DOWN Signal Accuracy: {down_signal_accuracy:.1f}% ({np.sum(strong_down_mask)} signals)")
    
    # Final Summary Section
    print(f"\n" + "="*70)
    print(f"🏆 FINAL PERFORMANCE SUMMARY")
    print(f"="*70)
    
    best_results = sorted_results[0][1]
    print(f"🥇 Best Model: {sorted_results[0][0]}")
    print(f"📊 Overall Accuracy: {best_results['accuracy']:.3f} ({best_results['accuracy']*100:.1f}%)")
    print(f"🎯 F1-Score (Weighted): {best_results['f1_weighted']:.3f}")
    
    if best_model_info:
        # Re-calculate key directional metrics for summary
        mask_up_down = (y_test != 0) & (best_model_info['predictions'] != 0)
        if mask_up_down.sum() > 0:
            directional_acc = accuracy_score(y_test[mask_up_down], best_model_info['predictions'][mask_up_down])
            print(f"📈 Directional Accuracy (Up/Down): {directional_acc:.3f} ({directional_acc*100:.1f}%)")
        
        # Signal accuracy
        up_signals = np.sum(best_model_info['predictions'] == 1)
        down_signals = np.sum(best_model_info['predictions'] == -1)
        neutral_signals = np.sum(best_model_info['predictions'] == 0)
        
        print(f"📡 Trading Signals Generated:")
        print(f"   UP signals: {up_signals}, DOWN signals: {down_signals}, NEUTRAL: {neutral_signals}")
        
        if up_signals > 0:
            up_accuracy = np.mean(y_test[best_model_info['predictions'] == 1] == 1) * 100
            print(f"   UP signal accuracy: {up_accuracy:.1f}%")
        if down_signals > 0:
            down_accuracy = np.mean(y_test[best_model_info['predictions'] == -1] == -1) * 100
            print(f"   DOWN signal accuracy: {down_accuracy:.1f}%")
    
    random_baseline = max(target_counts) / len(y)
    improvement = ((best_results['accuracy'] - random_baseline) / random_baseline) * 100
    
    if improvement > 0:
        print(f"✅ Beats random guessing by {improvement:.1f}%")
    else:
        print(f"⚠️  Only {improvement:.1f}% vs random guessing")
    
    print(f"="*70)
    
    # Trading Simulation Analysis
    print(f"\n💰 TRADING SIMULATION ANALYSIS")
    print(f"="*70)
    
    if best_model_info:
        # Get the actual returns for the test period (need next day returns)
        # We need to map test set back to original dataframe to get actual price movements
        test_dates = df_work.iloc[X_test.index]['date'].values
        test_targets = y_test.values
        predictions = best_model_info['predictions']
        
        # Get ACTUAL next-day returns from the feature matrix
        # Use the sol_actual_next_day_return column which contains the real market returns
        test_indices = X_test.index
        actual_returns = df_work.iloc[test_indices]['sol_actual_next_day_return'].values / 100  # Convert % to decimal
        
        # Handle any NaN values (last day won't have next day return)
        actual_returns = np.nan_to_num(actual_returns, nan=0.0)
        
        print(f"📊 Simulation Setup:")
        print(f"  Test period: {len(test_dates)} trading days")
        print(f"  Initial capital: $10,000")
        print(f"  Transaction cost: 0.1% per trade")
        print(f"  Using ACTUAL market returns (not simulated)")
        
        # Trading Strategy Simulation
        initial_capital = 10000
        capital = initial_capital
        position = 0  # 0 = cash, 1 = long, -1 = short
        transaction_cost = 0.001  # 0.1% transaction cost
        
        strategy_returns = []
        positions_held = []
        trades_made = 0
        
        for i, (prediction, actual_return) in enumerate(zip(predictions, actual_returns)):
            prev_position = position
            
            # Trading logic based on predictions
            if prediction == 1:  # Model says UP - go long
                if position != 1:
                    # Enter long position
                    if position == -1:  # Close short first
                        capital = capital * (1 - actual_return)  # Cover short
                        capital *= (1 - transaction_cost)  # Transaction cost
                    capital *= (1 - transaction_cost)  # Transaction cost for going long
                    position = 1
                    trades_made += 1
                # Hold long position
                day_return = actual_return
                
            elif prediction == -1:  # Model says DOWN - go short
                if position != -1:
                    # Enter short position  
                    if position == 1:  # Close long first
                        capital = capital * (1 + actual_return)  # Close long
                        capital *= (1 - transaction_cost)  # Transaction cost
                    capital *= (1 - transaction_cost)  # Transaction cost for going short
                    position = -1
                    trades_made += 1
                # Hold short position
                day_return = -actual_return  # Profit from price decline
                
            else:  # Model says NEUTRAL - stay in cash
                if position != 0:
                    # Close any position
                    if position == 1:
                        capital = capital * (1 + actual_return)  # Close long
                    else:  # position == -1
                        capital = capital * (1 - actual_return)  # Close short
                    capital *= (1 - transaction_cost)  # Transaction cost
                    position = 0
                    trades_made += 1
                day_return = 0  # Cash earns nothing
            
            # Apply the day's return to capital
            if position == 1:
                capital = capital * (1 + actual_return)
            elif position == -1:
                capital = capital * (1 - actual_return)
            # If position == 0 (cash), capital stays the same
            
            strategy_returns.append(capital / initial_capital - 1)  # Cumulative return
            positions_held.append(position)
        
        # Close final position
        if position != 0:
            capital *= (1 - transaction_cost)
        
        strategy_final_return = (capital - initial_capital) / initial_capital
        
        # Buy and Hold Strategy
        buy_hold_return = np.sum(actual_returns)  # Simple cumulative return
        buy_hold_final_value = initial_capital * (1 + buy_hold_return)
        
        # Calculate performance metrics
        strategy_returns = np.array(strategy_returns)
        daily_strategy_returns = np.diff(np.concatenate([[0], strategy_returns]))
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(daily_strategy_returns) > 1 and np.std(daily_strategy_returns) > 0:
            strategy_sharpe = np.mean(daily_strategy_returns) / np.std(daily_strategy_returns) * np.sqrt(252)
        else:
            strategy_sharpe = 0
            
        # Buy and hold Sharpe
        if len(actual_returns) > 1 and np.std(actual_returns) > 0:
            buy_hold_sharpe = np.mean(actual_returns) / np.std(actual_returns) * np.sqrt(252)
        else:
            buy_hold_sharpe = 0
        
        # Maximum drawdown for strategy
        cumulative_returns = 1 + strategy_returns
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Results
        print(f"\n🤖 ML STRATEGY RESULTS:")
        print(f"  Final Portfolio Value: ${capital:,.2f}")
        print(f"  Total Return: {strategy_final_return:.1%}")
        print(f"  Annualized Return: {(strategy_final_return * 252 / len(test_dates)):.1%}")
        print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"  Maximum Drawdown: {max_drawdown:.1%}")
        print(f"  Total Trades: {trades_made}")
        
        print(f"\n📈 BUY & HOLD BASELINE:")
        print(f"  Final Portfolio Value: ${buy_hold_final_value:,.2f}")
        print(f"  Total Return: {buy_hold_return:.1%}")
        print(f"  Annualized Return: {(buy_hold_return * 252 / len(test_dates)):.1%}")
        print(f"  Sharpe Ratio: {buy_hold_sharpe:.2f}")
        
        print(f"\n🏆 STRATEGY COMPARISON:")
        excess_return = strategy_final_return - buy_hold_return
        outperformance = ((capital / buy_hold_final_value) - 1) * 100
        
        if excess_return > 0:
            print(f"  ✅ ML Strategy BEATS Buy & Hold by {excess_return:.1%}")
            print(f"  🎯 Outperformance: {outperformance:.1f}%")
        else:
            print(f"  ❌ ML Strategy UNDERPERFORMS Buy & Hold by {abs(excess_return):.1%}")
            print(f"  📉 Underperformance: {abs(outperformance):.1f}%")
        
        # Trading signal effectiveness
        long_days = np.sum(np.array(positions_held) == 1)
        short_days = np.sum(np.array(positions_held) == -1)
        cash_days = np.sum(np.array(positions_held) == 0)
        
        print(f"\n📊 POSITION BREAKDOWN:")
        print(f"  Days Long: {long_days} ({long_days/len(positions_held)*100:.1f}%)")
        print(f"  Days Short: {short_days} ({short_days/len(positions_held)*100:.1f}%)")
        print(f"  Days Cash: {cash_days} ({cash_days/len(positions_held)*100:.1f}%)")
        
        # Calculate win rate
        profitable_trades = 0
        total_position_changes = 0
        
        for i in range(1, len(positions_held)):
            if positions_held[i] != positions_held[i-1]:  # Position change
                total_position_changes += 1
                # Check if the previous position was profitable
                if i > 1:
                    prev_return = actual_returns[i-1]
                    if positions_held[i-1] == 1 and prev_return > 0:  # Long position, positive return
                        profitable_trades += 1
                    elif positions_held[i-1] == -1 and prev_return < 0:  # Short position, negative return
                        profitable_trades += 1
        
        if total_position_changes > 0:
            win_rate = profitable_trades / total_position_changes
            print(f"  Trade Win Rate: {win_rate:.1%} ({profitable_trades}/{total_position_changes})")
    
    print(f"="*70)
    print(f"🎉 Advanced testing completed!")
    
    return results, best_models

if __name__ == "__main__":
    results, models = advanced_model_test()
