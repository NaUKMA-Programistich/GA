import random

A = 7.0
X_MIN = -5.12
X_MAX = 5.12
PRECISION = 2

DELTA = 0.01
SIGMA = 0.01

NR = 100
MAX_GENERATIONS = 1_000_000
MAX_GENERATIONS_STEADY_FACTOR = 1_000_000

MAX_GENERATIONS_STEADY_FACTOR = 1_00_000

CONVERGENCE_HOMOGENEITY_THRESHOLD = 0.99
CONVERGENCE_FITNESS_STABILITY_THRESHOLD = 0.0001
CONVERGENCE_FITNESS_STABILITY_WINDOW = 10

CONVERGENCE_FITNESS_STABILITY_WINDOW_STEADY_FACTOR = 10

POPULATION_SIZES = [100, 200, 300, 400]
DIMENSIONS = [1, 2, 3, 5]

CROSSOVER_PROBABILITIES = {
    n: [0.0, 0.6, 0.8, 1.0] for n in POPULATION_SIZES
}

MUTATION_PROBABILITIES = {
    100: [0.0, 0.001, 0.01, 0.1],
    200: [0.0, 0.0005, 0.005, 0.01, 0.1],
    300: [0.0, 0.0003, 0.003, 0.01, 0.1],
    400: [0.0, 0.0002, 0.0005, 0.002, 0.01, 0.1],
}

ENCODING_TYPES = ["binary", "gray"]
CROSSOVER_TYPES = ["one_point", "uniform"]
MUTATION_TYPES = ["density"]
REPRODUCTION_TYPES = ["generational", "steady_state"]

SELECTION_PARAMS_GENERATIONAL = {
    100: [
        "SUS", "RWS",
        "TournWITH_t=2", "TournWITHOUT_t=2", "TournWITHPART_t=2",
        "ExpRankRWS_c=0.9801", "ExpRankSUS_c=0.9801",
        "LinRankRWS_beta=2", "LinRankSUS_beta=2",
        "TournWITH_t=4", "TournWITHOUT_t=4",
        "ExpRankRWS_c=0.9606", "ExpRankSUS_c=0.9606",
        "LinRankRWS_beta=1.6", "LinRankSUS_beta=1.6"
    ],
    200: [
        "SUS", "RWS",
        "TournWITH_t=2", "TournWITHOUT_t=2", "TournWITHPART_t=2",
        "ExpRankRWS_c=0.99003", "ExpRankSUS_c=0.99003",
        "LinRankRWS_beta=2", "LinRankSUS_beta=2",
        "TournWITH_t=4", "TournWITHOUT_t=4",
        "ExpRankRWS_c=0.98015", "ExpRankSUS_c=0.98015",
        "LinRankRWS_beta=1.6", "LinRankSUS_beta=1.6"
    ],
    300: [
        "SUS", "RWS",
        "TournWITH_t=2", "TournWITHOUT_t=2", "TournWITHPART_t=2",
        "ExpRankRWS_c=0.99334", "ExpRankSUS_c=0.99334",
        "LinRankRWS_beta=2", "LinRankSUS_beta=2",
        "TournWITH_t=4", "TournWITHOUT_t=4",
        "ExpRankRWS_c=0.98673", "ExpRankSUS_c=0.98673",
        "LinRankRWS_beta=1.6", "LinRankSUS_beta=1.6"
    ],
    400: [
        "SUS", "RWS",
        "TournWITH_t=2", "TournWITHOUT_t=2", "TournWITHPART_t=2",
        "ExpRankRWS_c=0.99503", "ExpRankSUS_c=0.99503",
        "LinRankRWS_beta=2", "LinRankSUS_beta=2",
        "TournWITH_t=4", "TournWITHOUT_t=4",
        "ExpRankRWS_c=0.99004", "ExpRankSUS_c=0.99004",
        "LinRankRWS_beta=1.6", "LinRankSUS_beta=1.6"
    ]
}

SELECTION_METHODS_GENERATIONAL = SELECTION_PARAMS_GENERATIONAL

SELECTION_PARENT_STEADY = ["Elite", "RWS"]
SELECTION_SURVIVOR_STEADY = ["WorstComma", "RandComma", "WorstPlus", "RandPlus"]
GENERATION_GAPS = [0.05, 0.1, 0.2, 0.5]

DETAILED_STATS_FILENAME_TEMPLATE = "results/detailed_{repro_type}_dim_{dim}_pop_{pop_size}.csv"
SUMMARY_STATS_FILENAME_TEMPLATE = "results/summary_{repro_type}_dim_{dim}_pop_{pop_size}.csv"

OPTIMAL_X_PER_DIM = 0.0

GENE_LENGTH_N1 = 10
BINOMIAL_P = 0.5

def validate_constants():
    if X_MAX <= X_MIN:
        raise ValueError(f"X_MAX ({X_MAX}) має бути більше ніж X_MIN ({X_MIN})")
    
    if PRECISION <= 0:
        raise ValueError(f"PRECISION ({PRECISION}) має бути більше 0")
    
    for pop_size, probs in CROSSOVER_PROBABILITIES.items():
        invalid_probs = [p for p in probs if not 0 <= p <= 1]
        if invalid_probs:
            raise ValueError(f"Некоректні ймовірності кросинговеру для популяції {pop_size}: {invalid_probs}")
    
    for pop_size, probs in MUTATION_PROBABILITIES.items():
        invalid_probs = [p for p in probs if not 0 <= p <= 1]
        if invalid_probs:
            raise ValueError(f"Некоректні ймовірності мутації для популяції {pop_size}: {invalid_probs}")
    
    if not 0 <= BINOMIAL_P <= 1:
        raise ValueError(f"BINOMIAL_P ({BINOMIAL_P}) має бути в діапазоні [0,1]")

validate_constants()

def get_fixed_seeds(num_runs):
    random.seed(42)
    return [random.randint(0, 2**32 - 1) for _ in range(num_runs)]