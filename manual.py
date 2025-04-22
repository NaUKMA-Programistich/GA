import random
import numpy as np
import time
import argparse
import os
import math
import multiprocessing
from deap import base, tools, creator
from typing import Dict, Any, Tuple

import constants as C
import functions as F
import algorithms as A

os.makedirs("results", exist_ok=True)

try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

def assign_rank_probabilities(individuals, method='linear', param=None):
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness.values[0])
    n = len(sorted_individuals)
    
    if method == 'linear':
        beta = param if param is not None else 1.5
        if beta < 1.0 or beta > 2.0:
            raise ValueError("Параметр beta повинен бути між 1.0 та 2.0")
        
        alpha = 2.0 - beta
        probabilities = np.array([((alpha + (beta-alpha)*(i/(n-1))) if n > 1 else 1) for i in range(n)])
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.ones(n) / n
    
    elif method == 'exponential':
        c = param if param is not None else 0.975
        if c <= 0 or c >= 1:
            raise ValueError("Параметр c повинен бути між 0 та 1")
        
        probabilities = np.array([(1-c) * (c**i) for i in range(n)])
        
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.ones(n) / n
             
    else:
        raise ValueError("Невідомий метод ранжування")

    if not np.isclose(np.sum(probabilities), 1.0):
        probabilities = probabilities / np.sum(probabilities)

    return [probabilities[sorted_individuals.index(ind)] for ind in individuals]

def selRankSelection(individuals, k, method, param, use_sus):
    probabilities = assign_rank_probabilities(individuals, method=method, param=param)
    
    prob_sum = sum(probabilities)
    if not np.isclose(prob_sum, 1.0):
        probabilities = [p/prob_sum for p in probabilities]
    
    if use_sus:
        chosen = []
        distance = 1.0 / float(k)
        start = random.uniform(0, distance)
        points = [start + i * distance for i in range(k)]
        
        i = 0
        current_sum = probabilities[0]
        for p in points:
            while current_sum < p:
                i += 1
                if i >= len(individuals):
                    i = len(individuals) - 1
                    break
                current_sum += probabilities[i]
            chosen.append(individuals[i])
        return chosen
    else:
        chosen_indices = np.random.choice(len(individuals), size=k, replace=True, p=probabilities)
        return [individuals[i] for i in chosen_indices]

def selTournamentWithoutReplacement(individuals, k, tournsize):
    selected = []
    ind_copy = individuals.copy()
    
    while len(selected) < k:
        aspirant_count = min(tournsize, len(ind_copy))
        aspirants = random.sample(ind_copy, aspirant_count)
        
        winner = max(aspirants, key=lambda ind: ind.fitness.values[0])
        selected.append(winner)
        
        for aspirant in aspirants:
            ind_copy.remove(aspirant)
        
        if not ind_copy:
            ind_copy = individuals.copy()
    
    return selected[:k]

def selTournamentPartialReplacement(individuals, k, tournsize):
    selected = []
    winners = set()
    
    while len(selected) < k:
        remaining = [ind for i, ind in enumerate(individuals) if i not in winners]
        
        if not remaining:
            winners = set()
            remaining = individuals
            
        aspirants = random.sample(remaining, min(tournsize, len(remaining)))
        
        winner = max(aspirants, key=lambda ind: ind.fitness.values[0])
        selected.append(winner)
        
        winner_idx = individuals.index(winner)
        winners.add(winner_idx)
    
    return selected[:k]

def setup_toolbox(
    n_dims: int,
    pop_size: int,
    encoding_type: str,
    crossover_type: str,
    mutation_type: str,
    selection_method: str,
    reproduction_type: str,
    cx_pb: float,
    mut_pb: float,
    gene_length_per_dim: int = None,
) -> Tuple[base.Toolbox, int]:
    if gene_length_per_dim is None:
        gene_length_per_dim = F.calculate_gene_length(C.X_MIN, C.X_MAX, C.PRECISION)
    
    toolbox = base.Toolbox()
    
    total_gene_length = n_dims * gene_length_per_dim
    toolbox.register("individual", F.init_individual, creator.Individual, total_gene_length, C.BINOMIAL_P)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", F.evaluate, n_dims=n_dims, gene_length_per_dim=gene_length_per_dim, 
                    encoding_type=encoding_type, a=C.A, x_min=C.X_MIN, x_max=C.X_MAX)
    
    if reproduction_type == "generational":
        if selection_method.startswith("TournWITH_t="):
            t = int(selection_method.split("_t=")[1])
            toolbox.register("select", tools.selTournament, tournsize=t)
        elif selection_method.startswith("TournWITHOUT_t="):
            t = int(selection_method.split("_t=")[1])
            toolbox.register("select", selTournamentWithoutReplacement, tournsize=t)
        elif selection_method.startswith("TournWITHPART_t="):
            t = int(selection_method.split("_t=")[1])
            toolbox.register("select", selTournamentPartialReplacement, tournsize=t)
        elif selection_method == "SUS":
            toolbox.register("select", tools.selStochasticUniversalSampling)
        elif selection_method == "RWS":
            toolbox.register("select", tools.selRoulette)
        elif selection_method.startswith("ExpRankRWS_c="):
            c = float(selection_method.split("_c=")[1])
            toolbox.register("select", selRankSelection, method='exponential', param=c, use_sus=False)
        elif selection_method.startswith("ExpRankSUS_c="):
            c = float(selection_method.split("_c=")[1])
            toolbox.register("select", selRankSelection, method='exponential', param=c, use_sus=True)
        elif selection_method.startswith("LinRankRWS_beta="):
            beta = float(selection_method.split("_beta=")[1])
            toolbox.register("select", selRankSelection, method='linear', param=beta, use_sus=False)
        elif selection_method.startswith("LinRankSUS_beta="):
            beta = float(selection_method.split("_beta=")[1])
            toolbox.register("select", selRankSelection, method='linear', param=beta, use_sus=True)
        else:
            raise ValueError(f"Невідомий метод селекції: {selection_method} для генераційного типу")
    elif reproduction_type == "steady_state":
        pass
    else:
        raise ValueError(f"Невідомий тип репродукції: {reproduction_type}")
    
    if crossover_type == "one_point":
        toolbox.register("mate", tools.cxOnePoint)
    elif crossover_type == "uniform":
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    else:
        raise ValueError(f"Невідомий тип кросинговеру: {crossover_type}")
    
    if mutation_type == "density":
        indpb = mut_pb / total_gene_length if total_gene_length > 0 else 0
        toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    else:
        raise ValueError(f"Невідомий тип мутації: {mutation_type}")
    
    return toolbox, gene_length_per_dim

def run_generational_algorithm(
    n_dims: int,
    pop_size: int,
    encoding_type: str,
    crossover_type: str,
    mutation_type: str,
    selection_method: str,
    cx_pb: float,
    mut_pb: float,
    seed: int = None,
    verbose: bool = False
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    
    toolbox, gene_length_per_dim = setup_toolbox(
        n_dims=n_dims,
        pop_size=pop_size,
        encoding_type=encoding_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        selection_method=selection_method,
        reproduction_type="generational",
        cx_pb=cx_pb,
        mut_pb=mut_pb
    )
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    hof = tools.HallOfFame(1)
    
    pop = F.init_population_with_seed(toolbox, seed, pop_size)
    
    convergence_params = {
        'homogeneity_threshold': C.CONVERGENCE_HOMOGENEITY_THRESHOLD,
        'fitness_stability_threshold': C.CONVERGENCE_FITNESS_STABILITY_THRESHOLD,
        'fitness_stability_window': C.CONVERGENCE_FITNESS_STABILITY_WINDOW
    }
    
    start_time = time.time()
    final_pop, logbook, nfe = A.eaGenerational(
        population=pop,
        toolbox=toolbox,
        cxpb=cx_pb,
        mutpb=mut_pb,
        ngen=C.MAX_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
        convergence_params=convergence_params
    )
    end_time = time.time()
    runtime = end_time - start_time
    
    optimal_fitness = F.get_optimal_fitness(n_dims, C.A, C.OPTIMAL_X_PER_DIM)
    
    best_ind = hof[0]
    success, peak_accuracy, distance_accuracy = F.check_success(
        best_individual=best_ind,
        n_dims=n_dims,
        gene_length_per_dim=gene_length_per_dim,
        encoding_type=encoding_type,
        a=C.A,
        x_min=C.X_MIN,
        x_max=C.X_MAX,
        optimal_fitness=optimal_fitness,
        optimal_x_per_dim=C.OPTIMAL_X_PER_DIM,
        delta=C.DELTA,
        sigma=C.SIGMA
    )
    
    best_phenotype = F.decode_individual(best_ind, n_dims, gene_length_per_dim, C.X_MIN, C.X_MAX, encoding_type)
    
    fc = 0
    if logbook:
        best_fitness_history = [gen['max'] for gen in logbook]
        if len(best_fitness_history) >= 2:
            fitness_improvements = [max(0, best_fitness_history[i] - best_fitness_history[i-1]) for i in range(1, len(best_fitness_history))]
            total_improvement = best_fitness_history[-1] - best_fitness_history[0]
            if total_improvement > 0:
                fc = sum(improvement > 0 for improvement in fitness_improvements) / len(fitness_improvements)
            else:
                fc = 0
    
    results = {
        "success": success,
        "iterations": len(logbook),
        "nfe": nfe,
        "best_fitness": best_ind.fitness.values[0],
        "best_phenotype": best_phenotype,
        "avg_fitness": np.mean([ind.fitness.values[0] for ind in final_pop]),
        "peak_accuracy": peak_accuracy,
        "distance_accuracy": distance_accuracy,
        "fc": fc,
        "runtime": runtime,
        "logbook": logbook,
        "halloffame": hof
    }
    
    return results

def run_steady_state_algorithm(
    n_dims: int,
    pop_size: int,
    encoding_type: str,
    crossover_type: str,
    mutation_type: str,
    parent_selection_method: str,
    survivor_selection_method: str,
    generation_gap: float,
    cx_pb: float,
    mut_pb: float,
    seed: int = None,
    verbose: bool = False
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    toolbox, gene_length_per_dim = setup_toolbox(
        n_dims=n_dims,
        pop_size=pop_size,
        encoding_type=encoding_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        selection_method=None,
        reproduction_type="steady_state",
        cx_pb=cx_pb,
        mut_pb=mut_pb
    )
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    if parent_selection_method == "Elite":
        parent_selection_func = A.select_parents_elite
    elif parent_selection_method == "RWS":
        parent_selection_func = A.select_parents_rws
    else:
        raise ValueError(f"Невідомий метод селекції батьків: {parent_selection_method}")
    
    if survivor_selection_method == "WorstComma":
        survivor_selection_func = A.select_survivors_worst_comma
    elif survivor_selection_method == "RandComma":
        survivor_selection_func = A.select_survivors_rand_comma
    elif survivor_selection_method == "WorstPlus":
        survivor_selection_func = A.select_survivors_worst_plus
    elif survivor_selection_method == "RandPlus":
        survivor_selection_func = A.select_survivors_rand_plus
    else:
        raise ValueError(f"Невідомий метод селекції нащадків: {survivor_selection_method}")
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    hof = tools.HallOfFame(1)
    
    pop = toolbox.population(n=pop_size)
    
    max_nfe = int(generation_gap * C.MAX_GENERATIONS_STEADY_FACTOR)
    print(f"Running steady-state with max_nfe = {max_nfe} (GG={generation_gap}, Factor={C.MAX_GENERATIONS_STEADY_FACTOR})")

    convergence_params = {
        'homogeneity_threshold': C.CONVERGENCE_HOMOGENEITY_THRESHOLD,
        'fitness_stability_threshold': C.CONVERGENCE_FITNESS_STABILITY_THRESHOLD,
        'fitness_stability_window': math.ceil(C.CONVERGENCE_FITNESS_STABILITY_WINDOW / generation_gap) if generation_gap > 0 else C.CONVERGENCE_FITNESS_STABILITY_WINDOW
    }
    print(f"Steady-state convergence window: {convergence_params['fitness_stability_window']}")

    start_time = time.time()
    final_pop, logbook, nfe = A.eaSteadyState(
        population=pop,
        toolbox=toolbox,
        cxpb=cx_pb,
        mutpb=mut_pb,
        max_nfe=max_nfe,
        gg=generation_gap,
        parent_selection_func=parent_selection_func,
        survivor_selection_func=survivor_selection_func,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
        convergence_params=convergence_params
    )
    pool.close()
    end_time = time.time()
    runtime = end_time - start_time
    
    optimal_fitness = F.get_optimal_fitness(n_dims, C.A, C.OPTIMAL_X_PER_DIM)
    
    best_ind = hof[0] if len(hof) > 0 else None
    if best_ind is None:
        if final_pop:
            best_ind = tools.selBest(final_pop, 1)[0]
        else:
            print("Error: HOF and final population are empty.")
            return {
                "success": 0, "iterations": 0, "nfe": nfe, "best_fitness": None,
                "best_phenotype": None, "avg_fitness": None, "peak_accuracy": 0,
                "distance_accuracy": float('inf'), "fc": 0, "runtime": runtime, 
                "logbook": logbook, "halloffame": hof
            }

    success, peak_accuracy, distance_accuracy = F.check_success(
        best_individual=best_ind,
        n_dims=n_dims,
        gene_length_per_dim=gene_length_per_dim,
        encoding_type=encoding_type,
        a=C.A,
        x_min=C.X_MIN,
        x_max=C.X_MAX,
        optimal_fitness=optimal_fitness,
        optimal_x_per_dim=C.OPTIMAL_X_PER_DIM,
        delta=C.DELTA,
        sigma=C.SIGMA
    )
    
    best_phenotype = F.decode_individual(best_ind, n_dims, gene_length_per_dim, C.X_MIN, C.X_MAX, encoding_type)
    
    fc = 0
    if logbook:
        best_fitness_history = [gen['max'] for gen in logbook]
        if len(best_fitness_history) >= 2:
            fitness_improvements = [max(0, best_fitness_history[i] - best_fitness_history[i-1]) for i in range(1, len(best_fitness_history))]
            total_improvement = best_fitness_history[-1] - best_fitness_history[0]
            if total_improvement > 0:
                fc = sum(improvement > 0 for improvement in fitness_improvements) / len(fitness_improvements)
            else:
                fc = 0
    
    results = {
        "success": success,
        "iterations": len(logbook),
        "nfe": nfe,
        "best_fitness": best_ind.fitness.values[0],
        "best_phenotype": best_phenotype,
        "avg_fitness": np.mean([ind.fitness.values[0] for ind in final_pop if ind.fitness.valid]),
        "peak_accuracy": peak_accuracy,
        "distance_accuracy": distance_accuracy,
        "fc": fc,
        "runtime": runtime,
        "logbook": logbook,
        "halloffame": hof
    }
    
    return results

def display_results(
    results: Dict[str, Any],
    function_name: str = 'Unspecified',
    n_dims: int = None,
    pop_size: int = None,
    selection_reproduction_scheme: str = None,
    sampling_deletion_scheme: str = None,
    selection_param_gg: str = None,
    crossover_type: str = None,
    cx_pb: float = None,
    mutation_type: str = None,
    mut_pb: float = None,
    encoding_type: str = None,
    run_index: int = 1
) -> Dict[str, Any]:
    is_success = 1 if results['success'] else 0
    ni = results['iterations']
    nfe = results['nfe']
    
    if results['best_fitness'] is not None:
        f_max = f"{results['best_fitness']:.6f}"
    else:
        f_max = "None"
        
    if isinstance(results['best_phenotype'], list):
        x_max = '[' + ', '.join(f"{x:.4f}" for x in results['best_phenotype']) + ']'
    else:
        x_max = str(results['best_phenotype'])
    
    if results['avg_fitness'] is not None:
        f_avg = f"{results['avg_fitness']:.6f}"
    else:
        f_avg = "None"
    
    fc_value = results['fc'] if 'fc' in results else 0.0
    fc_display = f"{fc_value:.6f}"
    pa = f"{results['peak_accuracy']:.6f}"
    da = f"{results['distance_accuracy']:.6f}"
    runtime = f"{results['runtime']:.4f}"
    
    formatted_cx_pb = f"{cx_pb:.2f}" if cx_pb is not None else "N/A"
    formatted_mut_pb = f"{mut_pb:.2f}" if mut_pb is not None else "N/A"
    
    print(f"{'Успішно' if is_success else 'Неуспішно'}")
    print(f"Кількість записів у логбуці (ітерацій/замін): {ni}")
    print(f"Кількість обчислень функції фітнесу (NFE): {nfe}")
    print(f"Найкращий фітнес: {f_max}")
    print(f"Найкращий фенотип: {x_max}")
    print(f"Середній фітнес фінальної популяції: {f_avg}")
    print(f"Оцінка збіжності (FC): {fc_display}")
    print(f"Точність піку (PA): {pa}")
    print(f"Точність відстані (DA): {da}")
    print(f"Час виконання: {runtime}")
    
    return {
        "Function": function_name,
        "Dimension": n_dims,
        "Population": pop_size,
        "Selection/Reproduction scheme": selection_reproduction_scheme,
        "Sampling/Deletion scheme": sampling_deletion_scheme,
        "Selection param/GG": selection_param_gg,
        "Crossover": crossover_type,
        "Pc": formatted_cx_pb,
        "Mutation": mutation_type,
        "Pm": formatted_mut_pb,
        "Encoding": encoding_type,
        f"Run_{run_index}_IsSuc": is_success,
        f"Run_{run_index}_NI": ni,
        f"Run_{run_index}_NFE": nfe,
        f"Run_{run_index}_F_max": f_max,
        f"Run_{run_index}_x_max": x_max,
        f"Run_{run_index}_F_avg": f_avg,
        f"Run_{run_index}_FC": fc_value,
        f"Run_{run_index}_PA": results['peak_accuracy'],
        f"Run_{run_index}_DA": results['distance_accuracy']
    }

def run_algorithm_with_args(args):
    current_seed = args.seed
    print(f"Using seed: {current_seed}")
    random.seed(current_seed)
    np.random.seed(current_seed)

    if args.pop_size not in C.POPULATION_SIZES:
        print(f"Warning: Population size {args.pop_size} is not in the standard set {C.POPULATION_SIZES}")
        
    if args.cx_pb not in C.CROSSOVER_PROBABILITIES.get(args.pop_size, []):
        print(f"Warning: Crossover probability {args.cx_pb} is not in the standard set {C.CROSSOVER_PROBABILITIES.get(args.pop_size, [])}")
        
    if args.mut_pb not in C.MUTATION_PROBABILITIES.get(args.pop_size, []):
        print(f"Warning: No standard mutation probabilities for population size {args.pop_size}")

    function_name = "Rastrigin"
    
    if args.repro_type == "generational":
        selection_reproduction_scheme, sampling_deletion_scheme, selection_param_gg = F.parse_selection_scheme(args.selection)
        
        results = run_generational_algorithm(
            n_dims=args.dims,
            pop_size=args.pop_size,
            encoding_type=args.encoding,
            crossover_type=args.crossover,
            mutation_type=args.mutation,
            selection_method=args.selection,
            cx_pb=args.cx_pb,
            mut_pb=args.mut_pb,
            seed=current_seed,
            run_index=0,
            verbose=None
        )
    else: 
        selection_reproduction_scheme = args.parent_selection
        sampling_deletion_scheme = args.survivor_selection
        selection_param_gg = args.generation_gap
        
        results = run_steady_state_algorithm(
            n_dims=args.dims,
            pop_size=args.pop_size,
            encoding_type=args.encoding,
            crossover_type=args.crossover,
            mutation_type=args.mutation,
            parent_selection_method=args.parent_selection,
            survivor_selection_method=args.survivor_selection,
            generation_gap=args.generation_gap,
            cx_pb=args.cx_pb,
            mut_pb=args.mut_pb,
            seed=current_seed,
            verbose=None
        )
    
    result = display_results(
        results=results,
        function_name=function_name,
        n_dims=args.dims,
        pop_size=args.pop_size,
        selection_reproduction_scheme=selection_reproduction_scheme,
        sampling_deletion_scheme=sampling_deletion_scheme,
        selection_param_gg=selection_param_gg,
        crossover_type=args.crossover,
        cx_pb=args.cx_pb,
        mutation_type=args.mutation,
        mut_pb=args.mut_pb,
        encoding_type=args.encoding,
        run_index=1
    )
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Ручний запуск генетичного алгоритму")

    parser.add_argument("--dims", type=int, choices=C.DIMENSIONS, required=True, help="Розмірність")
    parser.add_argument("--pop_size", type=int, required=True, help="Розмір популяції")
    parser.add_argument("--encoding", type=str, choices=C.ENCODING_TYPES, required=True, help="Тип кодування")
    parser.add_argument("--crossover", type=str, choices=C.CROSSOVER_TYPES, required=True, help="Тип кросинговеру")
    parser.add_argument("--cx_pb", type=float, required=True, help="Ймовірність кросинговеру")
    parser.add_argument("--mutation", type=str, choices=C.MUTATION_TYPES, required=True, help="Тип мутації")
    parser.add_argument("--mut_pb", type=float, required=True, help="Ймовірність мутації")
    parser.add_argument("--repro_type", type=str, choices=C.REPRODUCTION_TYPES, required=True, help="Тип репродукції")
    parser.add_argument("--seed", type=int, required=True, help="Seed для відтворюваності")
    parser.add_argument("--selection", type=str, required=True, help="Метод селекції для генераційного типу")
    parser.add_argument("--parent_selection", type=str, choices=C.SELECTION_PARENT_STEADY, required=True, help="Метод селекції батьків для стійкого типу")
    parser.add_argument("--survivor_selection", type=str, choices=C.SELECTION_SURVIVOR_STEADY, required=True, help="Метод селекції нащадків для стійкого типу")
    parser.add_argument("--generation_gap", "--gg", type=float, choices=C.GENERATION_GAPS, required=True, help="Розрив покоління для стійкого типу")

    args = parser.parse_args()

    result = run_algorithm_with_args(args)
    
    for key, value in result.items():
        print(f"{key}=={value}")

if __name__ == "__main__":
    main()
