# Windows PowerShell script to run all tasks
# Task-specific total-timesteps: Humanoid=3M, Ant=3M, Others=1M

# Define tasks with their corresponding total-timesteps
$taskConfigs_ttpo = @(
    @{ Name = "Hopper-v4"; Steps = 1000000 },
    @{ Name = "Walker2d-v4"; Steps = 1000000 },
    @{ Name = "HalfCheetah-v4"; Steps = 1000000 },
    @{ Name = "Ant-v4"; Steps = 3000000 },
    @{ Name = "Humanoid-v4"; Steps = 3000000 }
)

Write-Host "Running opts_ttpo_continuous_action for all tasks..." -ForegroundColor Green
foreach ($config in $taskConfigs_ttpo) {
    $task = $config.Name
    $steps = $config.Steps
    Write-Host "Running task: $task ($($steps / 1000000)M steps)" -ForegroundColor Yellow
    python cleanrl/opts_ttpo_continuous_action.py `
        --env-id $task `
        --total-timesteps $steps `
        --num-steps 2048 `
        --num-envs 2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running task $task" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "Running opts_ttpo_continuous_action for all tasks..." -ForegroundColor Green
foreach ($config in $taskConfigs_ttpo) {
    $task = $config.Name
    $steps = $config.Steps
    Write-Host "Running task: $task ($($steps / 1000000)M steps)" -ForegroundColor Yellow
    python cleanrl/opts_ttpo_continuous_action.py `
        --env-id $task `
        --total-timesteps $steps `
        --num-steps 2048 `
        --num-envs 1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running task $task" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}


# $taskConfigs_ppo = @(
#     @{ Name = "HalfCheetah-v4"; Steps = 1000000 },
#     @{ Name = "Humanoid-v4"; Steps = 3000000 },
#     @{ Name = "Ant-v4"; Steps = 3000000 },
#     @{ Name = "Walker2d-v4"; Steps = 1000000 },
#     @{ Name = "Hopper-v4"; Steps = 1000000 }
# )

# Write-Host "`nRunning ppo_continuous_action for all tasks..." -ForegroundColor Green
# foreach ($config in $taskConfigs_ppo) {
#     $task = $config.Name
#     $steps = $config.Steps
#     Write-Host "Running task: $task ($($steps / 1000000)M steps)" -ForegroundColor Yellow
#     python cleanrl/ppo_continuous_action.py `
#         --env-id $task `
#         --total-timesteps $steps `
#         --num-steps 2048 `
#         --num-envs 1
#     if ($LASTEXITCODE -ne 0) {
#         Write-Host "Error running task $task" -ForegroundColor Red
#         exit $LASTEXITCODE
#     }
# }

Write-Host "`nAll tasks completed!" -ForegroundColor Green
