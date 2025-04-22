import math
import re
import numpy as np
from deap import creator, base
import constants as C
import random

FITNESS_CACHE_SIZE = 1024
fitness_cache = [(None, None)] * FITNESS_CACHE_SIZE

try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

def calculate_gene_length(x_min: float, x_max: float, precision: int) -> int:
    range_val = x_max - x_min
    num_steps = range_val * (10 ** precision)
    gene_length = math.ceil(math.log2(num_steps + 1))
    return gene_length

def get_optimal_fitness(n_dims: int, a: float, optimal_x: float) -> float:
    optimal_phenotype = [optimal_x] * n_dims
    return evaluate_target_function(optimal_phenotype, a)

def euclidean_distance(point1: list[float], point2: list[float]) -> float:
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    return math.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))


def decode_binary_segment(gene_segment: list[int], x_min: float, x_max: float, gene_length: int) -> float:
    max_int_value = (2 ** gene_length) - 1
    if max_int_value == 0:
        return x_min
    int_value = int("".join(map(str, gene_segment)), 2)
    float_value = x_min + (int_value / max_int_value) * (x_max - x_min)
    return min(float_value, x_max - 10**(-C.PRECISION - 1))

def decode_gray_segment(gene_segment: list[int], x_min: float, x_max: float, gene_length: int) -> float:
    binary_segment = [gene_segment[0]]
    for i in range(1, len(gene_segment)):
        binary_segment.append(binary_segment[i-1] ^ gene_segment[i])
    return decode_binary_segment(binary_segment, x_min, x_max, gene_length)

def decode_individual(individual: list[int], n_dims: int, gene_length_per_dim: int,
                      x_min: float, x_max: float, encoding_type: str) -> list[float]:
    phenotype = []
    start_index = 0
    for _ in range(n_dims):
        end_index = start_index + gene_length_per_dim
        gene_segment = individual[start_index:end_index]

        if encoding_type == "binary":
            value = decode_binary_segment(gene_segment, x_min, x_max, gene_length_per_dim)
        elif encoding_type == "gray":
            value = decode_gray_segment(gene_segment, x_min, x_max, gene_length_per_dim)
        else:
            raise ValueError(f"Невідомий тип кодування: {encoding_type}")

        phenotype.append(value)
        start_index = end_index

    if start_index != len(individual):
         raise ValueError(f"Довжина індивіда ({len(individual)}) не відповідає розмірності ({n_dims}) та довжині гена ({gene_length_per_dim})")

    return phenotype

def evaluate_target_function(phenotype: list[float], a: float) -> float:
    n = len(phenotype)
    if n == 0:
        raise ValueError("Розмірність популяції не може бути 0")

    constant_term = n * abs(10 * math.cos(2 * math.pi * a) - a**2)
    sum_term = sum(10 * math.cos(2 * math.pi * x_i) - x_i**2 for x_i in phenotype)
    return constant_term + sum_term

def hash_individual(individual: list[int]) -> int:
    individual_tuple = tuple(individual)
    hash_value = hash(individual_tuple)
    return hash_value % FITNESS_CACHE_SIZE

def evaluate(individual: list[int], n_dims: int, gene_length_per_dim: int,
             encoding_type: str, a: float, x_min: float, x_max: float) -> tuple[float]:
    
    cache_index = hash_individual(individual)
    cached_entry = fitness_cache[cache_index]

    if cached_entry[0] is not None and tuple(individual) == cached_entry[0]:
        return cached_entry[1]

    phenotype = decode_individual(individual, n_dims, gene_length_per_dim, x_min, x_max, encoding_type)
    fitness = evaluate_target_function(phenotype, a)
    fitness_tuple = (fitness,)

    fitness_cache[cache_index] = (tuple(individual), fitness_tuple)

    return fitness_tuple


def check_homogeneity(population: list, threshold: float) -> bool:
    if not population:
        return False

    num_individuals = len(population)
    if num_individuals < 2:
        return True

    gene_length = len(population[0])

    for j in range(gene_length):
        count_ones = sum(ind[j] for ind in population)
        proportion_ones = count_ones / num_individuals
        proportion_zeros = 1.0 - proportion_ones

        if max(proportion_ones, proportion_zeros) < threshold:
            return False

    return True

def check_fitness_stability(fitness_history: list[float], window_size: int, threshold: float) -> bool:
    if len(fitness_history) < window_size + 1:
        return False

    recent_avg_fitness = fitness_history[-(window_size + 1):]
    changes = [abs(recent_avg_fitness[i] - recent_avg_fitness[i-1]) for i in range(1, window_size + 1)]
    epsilon = 1e-10
    return all(change <= threshold + epsilon for change in changes)


def check_success(
        best_individual: creator.Individual,
        n_dims: int,
        gene_length_per_dim: int,
        encoding_type: str,
        a: float,
        x_min: float, x_max: float,
        optimal_fitness: float,
        optimal_x_per_dim: float,
        delta: float, sigma: float
    ) -> tuple[bool, float, float]:
    if not hasattr(best_individual, 'fitness') or not best_individual.fitness.valid:
         return False, 0.0, float('inf')

    best_fitness = best_individual.fitness.values[0]
    best_phenotype = decode_individual(best_individual, n_dims, gene_length_per_dim, x_min, x_max, encoding_type)

    fitness_achieved = best_fitness >= (1 - delta) * optimal_fitness
    peak_accuracy = best_fitness / optimal_fitness if optimal_fitness != 0 else (1.0 if best_fitness >= 0 else 0.0)


    optimal_point = [optimal_x_per_dim] * n_dims
    distance = euclidean_distance(best_phenotype, optimal_point)
    distance_achieved = distance <= sigma
    distance_accuracy = distance

    is_success = fitness_achieved and distance_achieved
    return is_success, peak_accuracy, distance_accuracy


def init_individual(icls: type, size: int, p: float) -> creator.Individual:
    genome = np.random.binomial(1, p, size=size).tolist()
    return icls(genome)

def init_population_with_seed(toolbox, seed, pop_size):
    random.seed(seed)
    np.random.seed(seed)
    return toolbox.population(n=pop_size)

def parse_selection_scheme(selection_scheme_name):
    if selection_scheme_name == 'SUS':
        return 'SUS', '-', '-'
    elif selection_scheme_name == 'RWS':
        return 'RWS', '-', '-'

    tourn_match = re.match(r"Tourn(WITH|WITHOUT|WITHPART)_t=(\d+)", selection_scheme_name)
    if tourn_match:
        tourn_type = tourn_match.group(1)
        param = f"t={tourn_match.group(2)}"
        scheme = f"Tourn"
        sampling = tourn_type
        return scheme, sampling, param

    exprank_match = re.match(r"ExpRank(RWS|SUS)_c=([\d.]+)", selection_scheme_name)
    if exprank_match:
        sampling = exprank_match.group(1)
        param = f"c={exprank_match.group(2)}"
        scheme = f"ExpRank"
        return scheme, sampling, param

    linrank_match = re.match(r"LinRank(RWS|SUS)_beta=([\d.]+)", selection_scheme_name)
    if linrank_match:
        sampling = linrank_match.group(1)
        param = f"beta={linrank_match.group(2)}"
        scheme = f"LinRank"
        return scheme, sampling, param

    return selection_scheme_name, '-', '-'
