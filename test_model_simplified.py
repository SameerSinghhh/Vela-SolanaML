#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
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
    exclude_cols = ['date', 'sol_close', 'sol_actual_next_day_return', 'btc_close', 'eth_close', 'target_next_day']
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
    
    # Prepare cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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
    
    print(f"📊 Simulation Setup:")
    print(f"  Period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    print(f"  Trading days: {len(test_dates)}")
    print(f"  Initial capital: $10,000")
    print(f"  Strategy: Long when predict UP (+1), Short when predict DOWN (-1)")
    print(f"  No transaction costs")
    print()
    
    # Initialize simulation
    initial_capital = 10000
    ml_capital = initial_capital
    buy_hold_capital = initial_capital
    
    # Track detailed results for CSV
    detailed_log = []
    
    print(f"📈 Day-by-Day Trading Results:")
    print(f"{'Date':<12} {'Pred':<4} {'Actual%':<8} {'Strategy%':<10} {'ML Value':<10} {'B&H Value':<10}")
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
        
        # Update portfolios
        ml_capital_before = ml_capital
        buy_hold_capital_before = buy_hold_capital
        
        ml_capital = ml_capital * (1 + strategy_return)
        buy_hold_capital = buy_hold_capital * (1 + actual_return)
        
        # Check if prediction is correct
        prediction_correct = (prediction == actual_target)
        
        # Log detailed data for CSV
        detailed_log.append({
            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
            'actual_target': int(actual_target),  # -1 or 1
            'prediction': int(prediction),  # -1 or 1
            'prediction_correct': prediction_correct,
            'actual_return_pct': actual_return * 100,
            'strategy_return_pct': strategy_return * 100,
            'ml_balance_before': ml_capital_before,
            'ml_balance_after': ml_capital,
            'buy_hold_balance_before': buy_hold_capital_before,
            'buy_hold_balance_after': buy_hold_capital
        })
        
        # Print every 10th day + first/last few days for display
        if i < 5 or i >= len(test_dates) - 5 or i % 10 == 0:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            pred_str = "UP" if prediction == 1 else "DOWN"
            actual_return_pct = actual_return * 100
            strategy_return_pct = strategy_return * 100
            print(f"{date_str:<12} {pred_str:<4} {actual_return_pct:>7.1f}% {strategy_return_pct:>9.1f}% ${ml_capital:>9.0f} ${buy_hold_capital:>10.0f}")
        elif i == 5:
            print(f"{'...':<12} {'...':<4} {'...':<8} {'...':<10} {'...':<10} {'...':<10}")
    
    # Save detailed CSV
    detailed_df = pd.DataFrame(detailed_log)
    detailed_df.to_csv('daily_trading_results.csv', index=False)
    print(f"\n✅ Detailed daily results saved to: daily_trading_results.csv")
    
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
    print()
    print(f"📈 Buy & Hold Strategy:")
    print(f"  Final Portfolio Value: ${buy_hold_capital:,.2f}")
    print(f"  Total Return: {buy_hold_total_return:.1%}")
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
    
    return results, best_models

if __name__ == "__main__":
    results, models = simplified_model_test() 