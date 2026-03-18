# Seeds run sequentially, tasks run in parallel (57 processes per round)

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
    SurroundNoFrameskip-v4
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

for seed in "${SEEDS[@]}"; do
    echo "OPTS_TTPO seed=$seed starting..."
    for task in "${ATARI_GAMES[@]}"; do
        python cleanrl/opts_ttpo_atari.py \
            --env-id $task \
            --total-timesteps 10000000 \
            --num-steps 4096 \
            --num-envs 1 \
            --no-cuda \
            --seed $seed &
    done
    wait
    echo "OPTS_TTPO seed=$seed done"
done