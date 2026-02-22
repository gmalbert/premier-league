"""
Validate that trained model files exist and load correctly.
Called from CI workflows after the training step.
"""
import os
import pickle

MODELS_DIR = 'models/'

required_models = [
    'xgb_baseline.pkl',
    'ensemble_model.pkl',
    'optimized_xgb.pkl',
    'model_performance.pkl',
]
optional_models = ['neural_model.pkl', 'neural_scaler.pkl']

missing_required = [m for m in required_models if not os.path.exists(os.path.join(MODELS_DIR, m))]
missing_optional = [m for m in optional_models if not os.path.exists(os.path.join(MODELS_DIR, m))]

if missing_required:
    raise FileNotFoundError(f'Missing required model files: {missing_required}')

if missing_optional:
    print(f'WARNING: Optional model files not found (this is OK): {missing_optional}')
else:
    print('All model files present (including optional neural network)')

with open(os.path.join(MODELS_DIR, 'model_performance.pkl'), 'rb') as f:
    perf = pickle.load(f)

print('Model performance loaded successfully')
for model_name, metrics in perf.items():
    if model_name == 'poisson':
        print(
            f"poisson: home_mae={metrics.get('home_mae', float('nan')):.3f}, "
            f"away_mae={metrics.get('away_mae', float('nan')):.3f}, "
            f"outcome_acc={metrics.get('outcome_acc', float('nan')):.3f}"
        )
    else:
        print(
            f"{model_name}: "
            f"Acc={metrics.get('accuracy', float('nan')):.3f}, "
            f"MAE={metrics.get('mae', float('nan')):.3f}"
        )
