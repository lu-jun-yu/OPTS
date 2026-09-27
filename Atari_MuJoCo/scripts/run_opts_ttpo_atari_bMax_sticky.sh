# OPTS-TTPO bMax + sticky actions (0.25). Seeds run sequentially, tasks run in parallel (57 processes per round)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3)

ATARI_GAMES=(
    AlienNoFrameskip-v4
    AmidarNoFrameskip-v4
    AssaultNoFrameskip-v4
    AsterixNoFrameskip-v4
    AsteroidsNoFrameskip-v4
    AtlantisNoFrameskip-v4
    BankHeistNoFrameskip-v4
    BattleZoneNoFrameskip-v4
    BeamRiderNoFrameskip-v4
    BerzerkNoFrameskip-v4
    BowlingNoFrameskip-v4
    BoxingNoFrameskip-v4
    BreakoutNoFrameskip-v4
    CentipedeNoFrameskip-v4
    ChopperCommandNoFrameskip-v4
    CrazyClimberNoFrameskip-v4
    DefenderNoFrameskip-v4
    DemonAttackNoFrameskip-v4
    DoubleDunkNoFrameskip-v4
    EnduroNoFrameskip-v4
    FishingDerbyNoFrameskip-v4
    FreewayNoFrameskip-v4
    FrostbiteNoFrameskip-v4
    GopherNoFrameskip-v4
    GravitarNoFrameskip-v4
    HeroNoFrameskip-v4
    IceHockeyNoFrameskip-v4
    JamesbondNoFrameskip-v4
    KangarooNoFrameskip-v4
    KrullNoFrameskip-v4
    KungFuMasterNoFrameskip-v4
    MontezumaRevengeNoFrameskip-v4
    MsPacmanNoFrameskip-v4
    NameThisGameNoFrameskip-v4
    PhoenixNoFrameskip-v4
    PitfallNoFrameskip-v4
    PongNoFrameskip-v4
    PrivateEyeNoFrameskip-v4
    QbertNoFrameskip-v4
    RiverraidNoFrameskip-v4
    RoadRunnerNoFrameskip-v4
    RobotankNoFrameskip-v4
    SeaquestNoFrameskip-v4
    SkiingNoFrameskip-v4
    SolarisNoFrameskip-v4
    SpaceInvadersNoFrameskip-v4
    StarGunnerNoFrameskip-v4
    ALE/Surround-v5
    TennisNoFrameskip-v4
    TimePilotNoFrameskip-v4
    TutankhamNoFrameskip-v4
    UpNDownNoFrameskip-v4
    VentureNoFrameskip-v4
    VideoPinballNoFrameskip-v4
    WizardOfWorNoFrameskip-v4
    YarsRevengeNoFrameskip-v4
    ZaxxonNoFrameskip-v4
)


EXP_NAME=opts_ttpo_atari_bMax_sticky
DATE_SUFFIX=20260917
NUM_ENVS=8
NUM_STEPS=128
XI=0.6
MAX_SEARCH=1

XI_FMT=$(python -c "print(float('${XI}'))")
ALGO="${EXP_NAME}_xi${XI_FMT}_s${MAX_SEARCH}_${DATE_SUFFIX}"
RESULTS_ROOT="/data/results/${NUM_ENVS}_${NUM_STEPS}"

skip_if_done() {
    # $1=algo dir, $2=task, $3=seed; "/" in env id becomes "_"
    local file="$RESULTS_ROOT/$1/${2//\//_}_$3.json"
    if [ -f "$file" ]; then
        echo "Skip (exists): $file"
        return 0
    fi
    return 1
}

for seed in "${SEEDS[@]}"; do
    echo "OPTS_TTPO seed=$seed starting..."
    for task in "${ATARI_GAMES[@]}"; do
        skip_if_done "$ALGO" "$task" "$seed" && continue
        python "cleanrl/cleanrl/${EXP_NAME}.py" \
            --env-id $task \
            --total-timesteps 10000000 \
            --num-steps $NUM_STEPS \
            --num-envs $NUM_ENVS \
            --xi $XI \
            --max-search-per-tree $MAX_SEARCH \
            --baseline mean \
            --no-cuda \
            --seed $seed &
    done
    wait
    echo "OPTS_TTPO seed=$seed done"
done
