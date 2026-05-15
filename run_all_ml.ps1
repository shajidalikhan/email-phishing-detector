$ErrorActionPreference = "Stop"
$venv_python = ".\venv\Scripts\python.exe"

$models = @("logistic_regression", "naive_bayes", "svm", "random_forest", "xgboost")

foreach ($model in $models) {
    Write-Host "========================================="
    Write-Host "Training $model..."
    & $venv_python "src\spam\models\ml\${model}_train.py"
    
    Write-Host "Evaluating $model..."
    & $venv_python "src\spam\models\ml\${model}_eval.py"
}

Write-Host "All ML models trained and evaluated successfully!"
