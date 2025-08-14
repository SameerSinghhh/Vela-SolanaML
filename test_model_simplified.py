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
import shap
import warnings
import json
warnings.filterwarnings('ignore')
import time

# 🔧 CONFIGURATION TOGGLE
# Set to False to run simulation WITHOUT transaction costs
ENABLE_TRANSACTION_COSTS = False

# 📊 VALIDATION SET CONFIGURATION
SPLITS = [
    (0.60, 0.20, 0.20),
    (0.70, 0.15, 0.15),
    (0.75, 0.15, 0.10),
    (0.80, 0.10, 0.10),
]

def chrono_slices(df, feature_cols, ratios):
    """Create chronological train/val/test splits"""
    n = len(df)
    tr, va, te = ratios
    i_tr = int(n * tr)
    i_va = int(n * (tr + va))
    train, val, test = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
    X_tr, y_tr = train[feature_cols], train['target_next_day']
    X_va, y_va = val[feature_cols], val['target_next_day']
    X_te, y_te = test[feature_cols], test['target_next_day']
    periods = {
        'train_period': (train['date'].min().date(), train['date'].max().date()),
        'val_period': (val['date'].min().date(), val['date'].max().date()),
        'test_period': (test['date'].min().date(), test['date'].max().date()),
    }
    return (X_tr, y_tr, X_va, y_va, X_te, y_te, periods)

def tune_and_select_on_val(X_tr, y_tr, X_va, y_va, models):
    """Tune hyperparameters on train and select best model on validation"""
    inner_cv = TimeSeriesSplit(n_splits=5)
    best = None
    
    for name, spec in models.items():
        print(f"    🔧 Tuning {name}...")
        
        # Create pipeline
        if name == 'SVM':
            pipe = Pipeline([('scaler', RobustScaler()), ('model', spec['model'])])
        else:
            pipe = Pipeline([('model', spec['model'])])

        # Handle XGBoost label mapping
        y_fit = y_tr.map({-1:0, 1:1}) if name == 'XGBoost' else y_tr
        
        # Grid search with time series CV
        gs = GridSearchCV(pipe, spec['params'], cv=inner_cv, scoring='f1_weighted', n_jobs=-1)
        gs.fit(X_tr, y_fit)

        # Predict on validation set
        if name == 'XGBoost':
            y_va_pred = pd.Series(gs.predict(X_va)).map({0:-1, 1:1}).values
        else:
            y_va_pred = gs.predict(X_va)

        # Calculate validation metrics
        val_f1w = f1_score(y_va, y_va_pred, average='weighted')
        val_acc = accuracy_score(y_va, y_va_pred)
        
        row = {
            'name': name, 
            'est': gs.best_estimator_, 
            'params': gs.best_params_,
            'cv_f1w': gs.best_score_, 
            'val_f1w': val_f1w, 
            'val_acc': val_acc
        }
        
        # Select best by validation F1 (tie-break by accuracy)
        if best is None or (val_f1w, val_acc) > (best['val_f1w'], best['val_acc']):
            best = row
            
        print(f"      ✅ CV F1: {gs.best_score_:.3f}, Val F1: {val_f1w:.3f}, Val Acc: {val_acc:.3f}")
    
    return best

def refit_trainval_and_test(best, X_tr, y_tr, X_va, y_va, X_te, y_te):
    """Refit best model on train+val and evaluate on test"""
    X_tv = pd.concat([X_tr, X_va])
    y_tv = pd.concat([y_tr, y_va])
    est = best['est']
    
    # Handle XGBoost label mapping
    if 'XGBoost' in best['name']:
        est.fit(X_tv, y_tv.map({-1:0, 1:1}))
        y_te_pred = pd.Series(est.predict(X_te)).map({0:-1, 1:1}).values
    else:
        est.fit(X_tv, y_tv)
        y_te_pred = est.predict(X_te)
        
    return {
        'test_acc': accuracy_score(y_te, y_te_pred),
        'test_f1w': f1_score(y_te, y_te_pred, average='weighted'),
        'y_test_pred': y_te_pred
    }

def simplified_model_test():
    """Simplified ML model testing with validation set approach and multiple splits"""
    
    print("🚀 MLSolana Simplified Model Test with Validation Set Approach")
    print("=" * 70)
    
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
    
    # Step 3: Define models with hyperparameter grids
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
    
    # Step 4: Evaluate multiple chronological splits with validation sets
    print(f"\n🔄 Evaluating multiple chronological splits with validation sets...")
    print("=" * 70)
    
    summary = []
    
    for i, ratios in enumerate(SPLITS):
        print(f"\n📊 SPLIT {i+1}: {ratios[0]*100:.0f}% Train / {ratios[1]*100:.0f}% Val / {ratios[2]*100:.0f}% Test")
        print("-" * 60)
        
        # Create chronological splits
        X_tr, y_tr, X_va, y_va, X_te, y_te, periods = chrono_slices(df, feature_cols, ratios)
        
        print(f"  📅 Train: {periods['train_period'][0]} to {periods['train_period'][1]} ({len(X_tr)} samples)")
        print(f"  📅 Val:   {periods['val_period'][0]} to {periods['val_period'][1]} ({len(X_va)} samples)")
        print(f"  📅 Test:  {periods['test_period'][0]} to {periods['test_period'][1]} ({len(X_te)} samples)")
        
        # Tune and select best model on validation set
        best = tune_and_select_on_val(X_tr, y_tr, X_va, y_va, models)
        
        # Refit on train+val and evaluate on test
        out = refit_trainval_and_test(best, X_tr, y_tr, X_va, y_va, X_te, y_te)
        
        # Store results
        summary.append({
            'split': f"{ratios[0]*100:.0f}/{ratios[1]*100:.0f}/{ratios[2]*100:.0f}",
            'chosen_model': best['name'],
            'cv_f1w': best['cv_f1w'],
            'val_f1w': best['val_f1w'],
            'val_acc': best['val_acc'],
            'test_f1w': out['test_f1w'],
            'test_acc': out['test_acc'],
            'best_params': best['params'],
            'train_period': f"{periods['train_period'][0]} to {periods['train_period'][1]}",
            'val_period': f"{periods['val_period'][0]} to {periods['val_period'][1]}",
            'test_period': f"{periods['test_period'][0]} to {periods['test_period'][1]}",
            'best_estimator': best['est'],
            'test_predictions': out['y_test_pred'],
            'test_data': pd.concat([X_te, y_te], axis=1)
        })
        
        print(f"  🏆 Best model: {best['name']} (Val F1: {best['val_f1w']:.3f}, Val Acc: {best['val_acc']:.3f})")
    
    # Create summary DataFrame and display results
    df_summary = pd.DataFrame(summary).sort_values('val_f1w', ascending=False)
    
    print(f"\n" + "=" * 70)
    print(f"📊 SPLIT SELECTION RESULTS (Sorted by Validation F1)")
    print("=" * 70)
    
    display_cols = ['split', 'chosen_model', 'cv_f1w', 'val_f1w', 'val_acc', 'test_f1w', 'test_acc']
    print(df_summary[display_cols].to_string(index=False))
    
    # Save results to CSV
    df_summary.to_csv('split_selection_results.csv', index=False)
    print(f"\n✅ Split selection results saved to: split_selection_results.csv")
    
    # Select best split by validation performance
    top = df_summary.iloc[0]
    print(f"\n🏆 SELECTED SPLIT: {top['split']} with model {top['chosen_model']}")
    print(f"   Validation F1: {top['val_f1w']:.3f}, Validation Accuracy: {top['val_acc']:.3f}")
    print(f"   Test F1: {top['test_f1w']:.3f}, Test Accuracy: {top['test_acc']:.3f}")
    
    # Extract best model and test data for downstream analysis
    best_model_name = top['chosen_model']
    best_model_obj = top['best_estimator']
    best_predictions = top['test_predictions']
    test_data = top['test_data']
    
    # Store results for compatibility with existing code
    results = {best_model_name: {
        'accuracy': top['test_acc'],
        'f1_weighted': top['test_f1w'],
        'cv_score': top['cv_f1w'],
        'predictions': best_predictions,
        'best_params': top['best_params']
    }}
    
    best_models = {best_model_name: best_model_obj}
    
    # Step 5: Display comprehensive results for selected model
    print(f"\n📊 COMPREHENSIVE MODEL RESULTS (Selected Split)")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Weighted':<12} {'CV Score':<10}")
    print("-" * 70)
    
    # Display results for the selected model
    for name, metrics in results.items():
        cv_display = f"{metrics['cv_score']:.3f}" if metrics['cv_score'] != 'N/A' else 'N/A'
        print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['f1_weighted']:<12.3f} {cv_display:<10}")
    
    # Step 6: Detailed analysis of best model
    best_model_name = list(results.keys())[0]
    best_metrics = results[best_model_name]
    best_predictions = best_metrics['predictions']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print("=" * 50)
    
    print(f"\n📈 Detailed Performance:")
    
    # Classification report - use test_data from the selected split
    y_test = test_data['target_next_day'].values
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
    
    # Comprehensive Feature Importance Analysis (Traditional + SHAP)
    print(f"\n" + "=" * 70)
    print(f"🎯 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print(f"=" * 70)
    
    best_model_obj = best_models[best_model_name]
    feature_names = [col for col in df.columns if col not in exclude_cols]
    
    # Traditional Feature Importance (if available)
    traditional_importance = None
    importance_type = None
    
    if hasattr(best_model_obj, 'feature_importances_'):
        # Tree-based models (RF, XGBoost, GB)
        traditional_importance = best_model_obj.feature_importances_
        importance_type = "Tree-based Feature Importance"
    elif hasattr(best_model_obj, 'coef_'):
        # Linear models (SVM with linear kernel)
        traditional_importance = np.abs(best_model_obj.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        # Try to get from pipeline
        if hasattr(best_model_obj.named_steps['model'], 'feature_importances_'):
            traditional_importance = best_model_obj.named_steps['model'].feature_importances_
            importance_type = "Tree-based Feature Importance"
        elif hasattr(best_model_obj.named_steps['model'], 'coef_'):
            traditional_importance = np.abs(best_model_obj.named_steps['model'].coef_[0])
            importance_type = "Coefficient Magnitude"
    
    # Display Traditional Importance (if available)
    if traditional_importance is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Traditional_Importance': traditional_importance
        }).sort_values('Traditional_Importance', ascending=False)
        
        print(f"📊 Traditional {importance_type}:")
        print(f"-" * 60)
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['Feature']:<35} {row['Traditional_Importance']:.4f}")
        print()
    
    # SHAP Analysis - Simple approach
    print(f"🔍 SHAP (SHapley Additive exPlanations) Analysis:")
    print(f"-" * 60)
    
    try:
        # Get train/test features as numpy arrays - use the selected split data
        # We need to get the training data from the selected split
        selected_split_idx = None
        for i, ratios in enumerate(SPLITS):
            if f"{ratios[0]*100:.0f}/{ratios[1]*100:.0f}/{ratios[2]*100:.0f}" == top['split']:
                selected_split_idx = i
                break
        
        if selected_split_idx is not None:
            # Recreate the selected split for SHAP analysis
            X_tr, y_tr, X_va, y_va, X_te, y_te, _ = chrono_slices(df, feature_cols, SPLITS[selected_split_idx])
            X_train_raw = pd.concat([X_tr, X_va])[feature_names].values  # Use train+val for background
            X_test_raw = X_te[feature_names].values
        else:
            # Fallback: use the test data we already have
            X_test_raw = test_data[feature_names].values
            # For background, we'll use a subset of the test data
            X_train_raw = X_test_raw[:100]  # Use first 100 test samples as background
        
        print(f"🔄 Computing SHAP values for {best_model_name}...")
        
        # Handle pipeline models - extract the actual model and apply scaling if needed
        if hasattr(best_model_obj, 'named_steps') and 'model' in best_model_obj.named_steps:
            # Extract the model and scaler if it exists
            if 'scaler' in best_model_obj.named_steps:
                scaler = best_model_obj.named_steps['scaler']
                model = best_model_obj.named_steps['model']
                
                # Apply scaling to data
                X_train = scaler.transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
            else:
                # No scaler, use model directly
                model = best_model_obj.named_steps['model']
                X_train = X_train_raw
                X_test = X_test_raw
        else:
            # For non-pipeline models
            model = best_model_obj
            X_train = X_train_raw
            X_test = X_test_raw
        
        # Use random sample of training data as background
        np.random.seed(42)  # For reproducible results
        background_indices = np.random.choice(len(X_train), size=100, replace=False)
        background_sample = X_train[background_indices]
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        
        # Get SHAP values for test sample
        test_sample = X_test[:10]  # First 10 test samples
        shap_values = explainer.shap_values(test_sample)
        
        # For binary classification, take the positive class (UP) - index 1
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_up = shap_values[1]  # UP class
        else:
            shap_values_up = shap_values
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.mean(np.abs(shap_values_up), axis=0)
        
        # Ensure arrays are 1-dimensional
        mean_shap_values = np.asarray(mean_shap_values).flatten()
        
        # Ensure arrays have same length (SHAP sometimes returns values for both classes)
        if len(feature_names) != len(mean_shap_values):
            min_len = min(len(feature_names), len(mean_shap_values))
            feature_names_adj = feature_names[:min_len]
            mean_shap_values = mean_shap_values[:min_len]
        else:
            feature_names_adj = feature_names
        
        # Create importance ranking
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names_adj,
            'SHAP_Importance': mean_shap_values
        }).sort_values('SHAP_Importance', ascending=False)
        
        print(f"📊 All Features by SHAP Importance:")
        print(f"-" * 50)
        for i, row in shap_importance_df.iterrows():
            print(f"{row['Feature']:<35} {row['SHAP_Importance']:.4f}")
        
        # Top insights
        top_3_shap = shap_importance_df.head(3)['Feature'].tolist()
        print(f"\n🥇 Top 3 Most Influential Features: {', '.join(top_3_shap)}")
        
        print(f"\n✅ SHAP analysis completed successfully!")
        
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {str(e)}")
        
        if traditional_importance is not None:
            print(f"\n💡 Key Insights from Traditional Importance:")
            top_3 = importance_df.head(3)['Feature'].tolist()
            print(f"  🥇 Top 3 features: {', '.join(top_3)}")
        else:
            print(f"⚠️  No feature importance analysis available for this model type")
    
    # Step 7: UP Bias Analysis
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
    
    # Step 8: Simple Trading Simulation
    print(f"\n" + "=" * 70)
    print(f"💰 SIMPLE TRADING SIMULATION")
    print(f"=" * 70)
    
    # Get test data for simulation - need to get the original dataframe data for the test period
    # Find the selected split and get the test period data from the original dataframe
    selected_split_idx = None
    for i, ratios in enumerate(SPLITS):
        if f"{ratios[0]*100:.0f}/{ratios[1]*100:.0f}/{ratios[2]*100:.0f}" == top['split']:
            selected_split_idx = i
            break
    
    if selected_split_idx is not None:
        # Recreate the selected split to get the test period from original dataframe
        _, _, _, _, _, _, periods = chrono_slices(df, feature_cols, SPLITS[selected_split_idx])
        test_start_date = periods['test_period'][0]
        test_end_date = periods['test_period'][1]
        
        # Get test period data from original dataframe
        test_mask = (df['date'].dt.date >= test_start_date) & (df['date'].dt.date <= test_end_date)
        test_period_data = df[test_mask].copy()
        
        test_dates = test_period_data['date'].values
        test_actual_returns = test_period_data['sol_actual_next_day_return'].values / 100  # Convert % to decimal
        test_actual_targets = test_period_data['target_next_day'].values  # Actual -1/1 classifications
    else:
        # Fallback: use the test_data we have
        test_dates = test_data.index.values  # Use index as dates
        test_actual_returns = np.zeros(len(test_data))  # Placeholder
        test_actual_targets = test_data['target_next_day'].values
    
    best_predictions = best_metrics['predictions']
    
    # Transaction cost parameters
    transaction_cost_per_trade = 0.002 if ENABLE_TRANSACTION_COSTS else 0.0  # 0.2% per round trip (0.1% buy + 0.1% sell)
    
    print(f"📊 Simulation Setup:")
    if selected_split_idx is not None:
        print(f"  Period: {test_start_date} to {test_end_date}")
    else:
        print(f"  Period: Test period from selected split")
    print(f"  Trading days: {len(test_dates)}")
    print(f"  Initial capital: $10,000")
    print(f"  Strategy: Long when predict UP (+1), Short when predict DOWN (-1)")
    if ENABLE_TRANSACTION_COSTS:
        print(f"  Transaction costs: {transaction_cost_per_trade*100:.1f}% per round trip (0.1% buy + 0.1% sell)")
    else:
        print(f"  Transaction costs: DISABLED (0.0%)")
    print()
    
    # Initialize simulation
    initial_capital = 10000
    ml_capital = initial_capital
    buy_hold_capital = initial_capital
    
    # Apply initial transaction cost for buy & hold strategy (0.1% to buy)
    buy_hold_capital = buy_hold_capital * (1 - transaction_cost_per_trade / 2)
    
    # Track detailed results for CSV
    detailed_log = []
    
    # Max drawdown tracking
    ml_peak = initial_capital
    buy_hold_peak = buy_hold_capital
    ml_max_drawdown = 0.0
    buy_hold_max_drawdown = 0.0
    
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
        
        # Update max drawdown tracking
        if ml_capital > ml_peak:
            ml_peak = ml_capital
        else:
            current_ml_drawdown = (ml_peak - ml_capital) / ml_peak
            ml_max_drawdown = max(ml_max_drawdown, current_ml_drawdown)
            
        if buy_hold_capital > buy_hold_peak:
            buy_hold_peak = buy_hold_capital
        else:
            current_bh_drawdown = (buy_hold_peak - buy_hold_capital) / buy_hold_peak
            buy_hold_max_drawdown = max(buy_hold_max_drawdown, current_bh_drawdown)
        
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
    
    # Step 9: Long-Only Strategy Variant
    print(f"\n" + "=" * 70)
    print(f"📈 LONG-ONLY STRATEGY SIMULATION")
    print(f"=" * 70)
    
    print(f"🔄 Calculating long-only strategy...")
    
    # Initialize long-only simulation
    long_only_capital = initial_capital
    long_only_peak = initial_capital
    long_only_max_drawdown = 0.0
    
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
        
        # Update max drawdown tracking for long-only
        if long_only_capital > long_only_peak:
            long_only_peak = long_only_capital
        else:
            current_lo_drawdown = (long_only_peak - long_only_capital) / long_only_peak
            long_only_max_drawdown = max(long_only_max_drawdown, current_lo_drawdown)
        
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
    print(f"  Max Drawdown: {ml_max_drawdown:.1%}")
    print(f"  Win Rate: {win_rate:.1%} ({winning_days}/{total_trading_days} days)")
    
    print(f"\n📈 Long-Only Strategy:")
    print(f"  Final Portfolio Value: ${long_only_capital:,.2f}")
    print(f"  Total Return: {long_only_total_return:.1%}")
    print(f"  Daily Mean Return: {long_only_mean_return:.4f} ({long_only_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {long_only_std_return:.4f} ({long_only_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio: {long_only_sharpe_annual:.3f}")
    print(f"  Max Drawdown: {long_only_max_drawdown:.1%}")
    print(f"  Long Positions: {len(long_positions)} days ({len(long_positions)/total_long_only_days*100:.1f}%) | Cash: {len(cash_positions)} days")
    
    print(f"\n📊 Buy & Hold Strategy:")
    print(f"  Final Portfolio Value: ${buy_hold_capital:,.2f}")
    print(f"  Total Return: {buy_hold_total_return:.1%}")
    print(f"  Daily Mean Return: {buy_hold_mean_return:.4f} ({buy_hold_mean_return*100:.2f}%)")
    print(f"  Daily Volatility: {buy_hold_std_return:.4f} ({buy_hold_std_return*100:.2f}%)")
    print(f"  Sharpe Ratio: {buy_hold_sharpe_annual:.3f}")
    print(f"  Max Drawdown: {buy_hold_max_drawdown:.1%}")
    
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
    print(f"\n💸 TRANSACTION COST IMPACT:")
    if ENABLE_TRANSACTION_COSTS:
        total_cost_pct = total_trades * transaction_cost_per_trade * 100
        print(f"  Total trades: {total_trades} | Cost per trade: {transaction_cost_per_trade*100:.1f}% | Total cost: {total_cost_pct:.1f}%")
    else:
        print(f"  Transaction costs: DISABLED (0.0%) - Running simulation without costs")
    
    print(f"\n✅ Complete trading data saved to: daily_trading_results.csv")
    print(f"🎉 Simulation completed - {len(detailed_log)} trading days analyzed")
    
    return results, best_models

if __name__ == "__main__":
    results, models = simplified_model_test() 