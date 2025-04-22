import os
import argparse
import numpy as np
import time
import pandas as pd
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args
from deap import creator, base
import multiprocessing
import concurrent.futures

import constants as C
from manual import run_generational_algorithm, run_steady_state_algorithm

try:
    del creator.FitnessMax
    del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

os.makedirs("results/smbo", exist_ok=True)

def run_single_generational(n_dims, pop_size, encoding_type, crossover_type, mutation_type,
                           selection_method, cx_pb, mut_pb, seed, verbose=False):
    try:
        if verbose:
            print(f"Прогін з seed={seed}: n_dims={n_dims}, pop_size={pop_size}, "
                 f"encoding={encoding_type}, crossover={crossover_type}, mutation={mutation_type}, "
                 f"selection={selection_method}, cx_pb={cx_pb}, mut_pb={mut_pb}")
        
        result = run_generational_algorithm(
            n_dims=n_dims,
            pop_size=pop_size,
            encoding_type=encoding_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            selection_method=selection_method,
            cx_pb=cx_pb,
            mut_pb=mut_pb,
            seed=seed,
            verbose=False
        )
        return result
    except Exception as e:
        print(f"Помилка при запуску алгоритму: {e}")
        return None

def run_single_steady_state(n_dims, pop_size, encoding_type, crossover_type, mutation_type,
                           parent_selection, survivor_selection, generation_gap,
                           cx_pb, mut_pb, seed, verbose=False):
    try:
        if verbose:
            print(f"Прогін з seed={seed}: n_dims={n_dims}, pop_size={pop_size}, "
                 f"encoding={encoding_type}, crossover={crossover_type}, mutation={mutation_type}, "
                 f"parent_sel={parent_selection}, surv_sel={survivor_selection}, "
                 f"gg={generation_gap}, cx_pb={cx_pb}, mut_pb={mut_pb}")
        
        result = run_steady_state_algorithm(
            n_dims=n_dims,
            pop_size=pop_size,
            encoding_type=encoding_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            parent_selection_method=parent_selection,
            survivor_selection_method=survivor_selection,
            generation_gap=generation_gap,
            cx_pb=cx_pb,
            mut_pb=mut_pb,
            seed=seed,
            verbose=False
        )
        return result
    except Exception as e:
        print(f"Помилка при запуску алгоритму: {e}")
        return None

def run_with_params(params, n_dims, pop_size, repro_type, nr_runs=5, verbose=False, max_workers=None):
    seeds = C.get_fixed_seeds(nr_runs)
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        if repro_type == "generational":
            encoding_type, crossover_type, mutation_type, cx_pb, mut_pb, selection_method = params
            
            for seed in seeds:
                future = executor.submit(
                    run_single_generational,
                    n_dims=n_dims,
                    pop_size=pop_size,
                    encoding_type=encoding_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    selection_method=selection_method,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    seed=seed,
                    verbose=verbose
                )
                futures.append(future)
        else:
            encoding_type, crossover_type, mutation_type, parent_selection, survivor_selection, generation_gap, cx_pb, mut_pb = params
            
            for seed in seeds:
                future = executor.submit(
                    run_single_steady_state,
                    n_dims=n_dims,
                    pop_size=pop_size,
                    encoding_type=encoding_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    parent_selection=parent_selection,
                    survivor_selection=survivor_selection,
                    generation_gap=generation_gap,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    seed=seed,
                    verbose=verbose
                )
                futures.append(future)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
            if verbose:
                print(f"Завершено прогін {i+1}/{nr_runs}")
    
    successes = sum(result.get('success', False) for result in results)
    success_rate = successes / len(results) if results else 0
    
    if success_rate > 0:
        successful_results = [r for r in results if r.get('success', False)]
        avg_nfe = np.mean([r.get('nfe', float('inf')) for r in successful_results])
        avg_peak_accuracy = np.mean([r.get('peak_accuracy', 0) for r in successful_results])
        
        if repro_type == "generational":
            max_allowed_nfe = C.MAX_GENERATIONS
        else:
            generation_gap = params[5]
            max_allowed_nfe = generation_gap * C.MAX_GENERATIONS_STEADY_FACTOR
            
        normalized_nfe = avg_nfe / max_allowed_nfe if max_allowed_nfe > 0 else 1
        
        return -1 * (success_rate - 0.1 * normalized_nfe + 0.1 * avg_peak_accuracy)
    else:
        return 0

def setup_generational_space(pop_size):
    return [
        Categorical(C.ENCODING_TYPES, name='encoding_type'),
        Categorical(C.CROSSOVER_TYPES, name='crossover_type'),
        Categorical(C.MUTATION_TYPES, name='mutation_type'),
        Categorical(C.CROSSOVER_PROBABILITIES[pop_size], name='cx_pb'),
        Categorical(C.MUTATION_PROBABILITIES[pop_size], name='mut_pb'),
        Categorical(C.SELECTION_PARAMS_GENERATIONAL[pop_size], name='selection_method')
    ]

def setup_steady_state_space(pop_size):
    return [
        Categorical(C.ENCODING_TYPES, name='encoding_type'),
        Categorical(C.CROSSOVER_TYPES, name='crossover_type'),
        Categorical(C.MUTATION_TYPES, name='mutation_type'),
        Categorical(C.SELECTION_PARENT_STEADY, name='parent_selection'),
        Categorical(C.SELECTION_SURVIVOR_STEADY, name='survivor_selection'),
        Categorical(C.GENERATION_GAPS, name='generation_gap'),
        Categorical(C.CROSSOVER_PROBABILITIES[pop_size], name='cx_pb'),
        Categorical(C.MUTATION_PROBABILITIES[pop_size], name='mut_pb')
    ]

def run_smbo_generational(n_dims, pop_size, n_calls=30, nr_runs=5, verbose=False, max_workers=None):
    space = setup_generational_space(pop_size)
    file_lock = multiprocessing.Manager().Lock()
    
    @use_named_args(space)
    def objective(**params):
        param_list = [
            params['encoding_type'],
            params['crossover_type'],
            params['mutation_type'],
            params['cx_pb'],
            params['mut_pb'],
            params['selection_method']
        ]
        
        if verbose:
            print(f"Оцінюємо конфігурацію: {params}")
            
        return run_with_params(param_list, n_dims, pop_size, "generational", nr_runs, verbose, max_workers)
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=verbose
    )
    
    best_params = {dim.name: value for dim, value in zip(space, result.x)}
    best_metric = -result.fun
    
    filename = f"results/smbo/generational_dim{n_dims}_pop{pop_size}.csv"
    
    params_df = pd.DataFrame({
        'dimension': [n_dims],
        'population': [pop_size],
        'encoding_type': [best_params['encoding_type']],
        'crossover_type': [best_params['crossover_type']],
        'mutation_type': [best_params['mutation_type']],
        'cx_pb': [best_params['cx_pb']],
        'mut_pb': [best_params['mut_pb']],
        'selection_method': [best_params['selection_method']],
        'metric': [best_metric]
    })
    
    with file_lock:
        params_df.to_csv(filename, index=False)
    
    print(f"Найкращі параметри для n_dims={n_dims}, pop_size={pop_size}:")
    for name, value in best_params.items():
        print(f"{name}: {value}")
    print(f"Метрика: {best_metric}")
    
    return best_params, best_metric

def run_smbo_steady_state(n_dims, pop_size, n_calls=30, nr_runs=5, verbose=False, max_workers=None):
    space = setup_steady_state_space(pop_size)
    file_lock = multiprocessing.Manager().Lock()
    
    @use_named_args(space)
    def objective(**params):
        param_list = [
            params['encoding_type'],
            params['crossover_type'],
            params['mutation_type'],
            params['parent_selection'],
            params['survivor_selection'],
            params['generation_gap'],
            params['cx_pb'],
            params['mut_pb']
        ]
        
        if verbose:
            print(f"Оцінюємо конфігурацію: {params}")
            
        return run_with_params(param_list, n_dims, pop_size, "steady_state", nr_runs, verbose, max_workers)
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=verbose
    )
    
    best_params = {dim.name: value for dim, value in zip(space, result.x)}
    best_metric = -result.fun
    
    filename = f"results/smbo/steady_state_dim{n_dims}_pop{pop_size}.csv"
    
    params_df = pd.DataFrame({
        'dimension': [n_dims],
        'population': [pop_size],
        'encoding_type': [best_params['encoding_type']],
        'crossover_type': [best_params['crossover_type']],
        'mutation_type': [best_params['mutation_type']],
        'parent_selection': [best_params['parent_selection']],
        'survivor_selection': [best_params['survivor_selection']],
        'generation_gap': [best_params['generation_gap']],
        'cx_pb': [best_params['cx_pb']],
        'mut_pb': [best_params['mut_pb']],
        'metric': [best_metric]
    })
    
    with file_lock:
        params_df.to_csv(filename, index=False)
    
    print(f"Найкращі параметри для n_dims={n_dims}, pop_size={pop_size}:")
    for name, value in best_params.items():
        print(f"{name}: {value}")
    print(f"Метрика: {best_metric}")
    
    return best_params, best_metric

def validate_best_params(params, n_dims, pop_size, repro_type, nr_runs=30, verbose=False, max_workers=None):
    file_lock = multiprocessing.Manager().Lock()
    seeds = C.get_fixed_seeds(nr_runs)
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        if repro_type == "generational":
            encoding_type, crossover_type, mutation_type, cx_pb, mut_pb, selection_method = params
            
            for seed in seeds:
                future = executor.submit(
                    run_single_generational,
                    n_dims=n_dims,
                    pop_size=pop_size,
                    encoding_type=encoding_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    selection_method=selection_method,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    seed=seed,
                    verbose=verbose
                )
                futures.append(future)
        else:
            encoding_type, crossover_type, mutation_type, parent_selection, survivor_selection, generation_gap, cx_pb, mut_pb = params
            
            for seed in seeds:
                future = executor.submit(
                    run_single_steady_state,
                    n_dims=n_dims,
                    pop_size=pop_size,
                    encoding_type=encoding_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    parent_selection=parent_selection,
                    survivor_selection=survivor_selection,
                    generation_gap=generation_gap,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    seed=seed,
                    verbose=verbose
                )
                futures.append(future)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
            if verbose:
                print(f"Завершено валідацію {i+1}/{nr_runs}")
    
    successes = sum(result.get('success', False) for result in results)
    success_rate = successes / len(results) if results else 0
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        avg_nfe = np.mean([r.get('nfe', 0) for r in successful_results])
        avg_iterations = np.mean([r.get('iterations', 0) for r in successful_results])
        avg_peak_accuracy = np.mean([r.get('peak_accuracy', 0) for r in successful_results])
        avg_distance_accuracy = np.mean([r.get('distance_accuracy', 0) for r in successful_results])
        avg_runtime = np.mean([r.get('runtime', 0) for r in successful_results])
    else:
        avg_nfe = 0
        avg_iterations = 0
        avg_peak_accuracy = 0
        avg_distance_accuracy = 0
        avg_runtime = 0
    
    validation_results = {
        'success_rate': success_rate,
        'avg_nfe': avg_nfe,
        'avg_iterations': avg_iterations,
        'avg_peak_accuracy': avg_peak_accuracy,
        'avg_distance_accuracy': avg_distance_accuracy,
        'avg_runtime': avg_runtime
    }
    
    print(f"Результати валідації для n_dims={n_dims}, pop_size={pop_size}:")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Avg NFE: {avg_nfe:.2f}")
    print(f"Avg iterations: {avg_iterations:.2f}")
    print(f"Avg peak accuracy: {avg_peak_accuracy:.4f}")
    print(f"Avg distance accuracy: {avg_distance_accuracy:.4f}")
    print(f"Avg runtime: {avg_runtime:.2f}s")
    
    validation_filename = f"results/smbo/{repro_type}_validation_dim{n_dims}_pop{pop_size}.csv"
    
    if repro_type == "generational":
        validation_df = pd.DataFrame({
            'dimension': [n_dims],
            'population': [pop_size],
            'encoding_type': [encoding_type],
            'crossover_type': [crossover_type],
            'mutation_type': [mutation_type],
            'cx_pb': [cx_pb],
            'mut_pb': [mut_pb],
            'selection_method': [selection_method],
            'success_rate': [success_rate],
            'avg_nfe': [avg_nfe],
            'avg_iterations': [avg_iterations],
            'avg_peak_accuracy': [avg_peak_accuracy],
            'avg_distance_accuracy': [avg_distance_accuracy],
            'avg_runtime': [avg_runtime]
        })
    else:
        validation_df = pd.DataFrame({
            'dimension': [n_dims],
            'population': [pop_size],
            'encoding_type': [encoding_type],
            'crossover_type': [crossover_type],
            'mutation_type': [mutation_type],
            'parent_selection': [parent_selection],
            'survivor_selection': [survivor_selection],
            'generation_gap': [generation_gap],
            'cx_pb': [cx_pb],
            'mut_pb': [mut_pb],
            'success_rate': [success_rate],
            'avg_nfe': [avg_nfe],
            'avg_iterations': [avg_iterations],
            'avg_peak_accuracy': [avg_peak_accuracy],
            'avg_distance_accuracy': [avg_distance_accuracy],
            'avg_runtime': [avg_runtime]
        })
    
    with file_lock:
        validation_df.to_csv(validation_filename, index=False)
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Автоматичний підбір параметрів ГА за допомогою SMBO")
    
    parser.add_argument("--dims", type=int, required=True, choices=C.DIMENSIONS, help="Розмірність задачі")
    parser.add_argument("--pop_size", type=int, required=True, choices=C.POPULATION_SIZES, help="Розмір популяції")
    parser.add_argument("--repro_type", type=str, required=True, choices=C.REPRODUCTION_TYPES, help="Тип репродукції")
    parser.add_argument("--n_calls", type=int, default=30, help="Кількість ітерацій SMBO")
    parser.add_argument("--n_runs", type=int, default=5, help="Кількість прогонів для кожного набору параметрів")
    parser.add_argument("--validate", action="store_true", help="Валідувати найкращі параметри")
    parser.add_argument("--validate_runs", type=int, default=30, help="Кількість прогонів для валідації")
    parser.add_argument("--verbose", action="store_true", help="Детальний вивід")
    parser.add_argument("--max_workers", type=int, default=None, help="Максимальна кількість процесів")
    
    args = parser.parse_args()
    
    print(f"Запуск SMBO для dims={args.dims}, pop_size={args.pop_size}, repro_type={args.repro_type}")
    print(f"Кількість ітерацій SMBO: {args.n_calls}")
    print(f"Кількість прогонів для кожного набору параметрів: {args.n_runs}")
    
    start_time = time.time()
    
    if args.repro_type == "generational":
        best_params, best_metric = run_smbo_generational(
            n_dims=args.dims,
            pop_size=args.pop_size,
            n_calls=args.n_calls,
            nr_runs=args.n_runs,
            verbose=args.verbose,
            max_workers=args.max_workers
        )
        
        param_list = [
            best_params['encoding_type'],
            best_params['crossover_type'],
            best_params['mutation_type'],
            best_params['cx_pb'],
            best_params['mut_pb'],
            best_params['selection_method']
        ]
    else:
        best_params, best_metric = run_smbo_steady_state(
            n_dims=args.dims,
            pop_size=args.pop_size,
            n_calls=args.n_calls,
            nr_runs=args.n_runs,
            verbose=args.verbose,
            max_workers=args.max_workers
        )
        
        param_list = [
            best_params['encoding_type'],
            best_params['crossover_type'],
            best_params['mutation_type'],
            best_params['parent_selection'],
            best_params['survivor_selection'],
            best_params['generation_gap'],
            best_params['cx_pb'],
            best_params['mut_pb']
        ]
    
    if args.validate:
        print(f"Валідація найкращих параметрів ({args.validate_runs} прогонів)...")
        validate_best_params(
            params=param_list,
            n_dims=args.dims,
            pop_size=args.pop_size,
            repro_type=args.repro_type,
            nr_runs=args.validate_runs,
            verbose=args.verbose,
            max_workers=args.max_workers
        )
    
    print(f"Загальний час виконання: {(time.time() - start_time) / 60:.2f} хвилин")

if __name__ == "__main__":
    main()
