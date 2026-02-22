perf={'xgb_baseline':{'accuracy':0.7,'mae':0.3},'poisson':{'home_mae':0.5,'away_mae':0.6,'outcome_acc':0.4}}
print('Simulating output:')
for model_name, metrics in perf.items():
    if model_name=='poisson':
        print(f"poisson: home_mae={metrics.get('home_mae', float('nan')):.3f}, away_mae={metrics.get('away_mae', float('nan')):.3f}, outcome_acc={metrics.get('outcome_acc', float('nan')):.3f}")
    else:
        print(f"{model_name}: Acc={metrics.get('accuracy', float('nan')):.3f}, MAE={metrics.get('mae', float('nan')):.3f}")
