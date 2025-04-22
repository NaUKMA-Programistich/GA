import argparse
import itertools
import os
import concurrent.futures
import multiprocessing
import time

from deap import base, creator

import constants as C
from manual import run_generational_algorithm, run_steady_state_algorithm
from stats import prepare_detail_file, prepare_summary_file, write_to_files

try:
    del creator.FitnessMax
    del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

os.makedirs("results", exist_ok=True)

def run_generational_configuration(
        n_dims,
        pop_size,
        encoding_type,
        crossover_type,
        mutation_type,
        selection_method,
        cx_pb, mut_pb,
        nr_runs,
        file_lock=None
):
    seeds = C.get_fixed_seeds(nr_runs)
    results = []
    for i, seed in enumerate(seeds):
        try:
            print(f"Прогін {i+1}/{nr_runs}: n_dims={n_dims}, pop_size={pop_size}, "
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
            results.append(result)
        
        except Exception as e:
            print(f"Помилка при запуску алгоритму: {e}")

    if file_lock:
        with file_lock:
            write_to_files(
                repro_type="generational",
                dim=n_dims,
                pop_size=pop_size,
                count_runs=nr_runs,
                encoding_type=encoding_type,
                crossover_type=crossover_type,
                mutation_type=mutation_type,
                selection_method=selection_method,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                results=results
            )
    else:
        write_to_files(
            repro_type="generational",
            dim=n_dims,
            pop_size=pop_size,
            count_runs=nr_runs,
            encoding_type=encoding_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            selection_method=selection_method,
            cx_pb=cx_pb,
            mut_pb=mut_pb,
            results=results
        )

def run_steady_state_configuration(
        n_dims, pop_size, encoding_type,
        crossover_type,
        mutation_type, parent_selection,
        survivor_selection,
        generation_gap,
        cx_pb,
        mut_pb,
        nr_runs,
        file_lock=None
):
    seeds = C.get_fixed_seeds(nr_runs)
    results = []
    for i, seed in enumerate(seeds):
        try:
            print(f"Прогін {i+1}/{nr_runs}: n_dims={n_dims}, pop_size={pop_size}, "
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
            results.append(result)
                
        except Exception as e:
            print(f"Помилка при запуску алгоритму: {e}")
    
    if file_lock:
        with file_lock:
            write_to_files(
                repro_type="steady_state",
                dim=n_dims,
                pop_size=pop_size,
                count_runs=nr_runs,
                results=results
            )
    else:
        write_to_files(
            repro_type="steady_state",
            dim=n_dims,
            pop_size=pop_size,
            count_runs=nr_runs,
            results=results
        )

def run_generational_test(n_dims, pop_size, nr_runs):
    encoding_types = C.ENCODING_TYPES
    crossover_types = C.CROSSOVER_TYPES
    mutation_types = C.MUTATION_TYPES
    crossover_probs = C.CROSSOVER_PROBABILITIES[pop_size]
    mutation_probs = C.MUTATION_PROBABILITIES[pop_size]
    selection_methods = C.SELECTION_METHODS_GENERATIONAL[pop_size]

    total_configs = (len(encoding_types) * len(crossover_types) * len(mutation_types) * 
                     len(crossover_probs) * len(mutation_probs) * len(selection_methods))
    
    print(f"Запускаємо {total_configs} конфігурацій для генераційного типу")
    
    file_lock = multiprocessing.Manager().Lock()
    
    configs = []
    for encoding, crossover, mutation, cx_pb, mut_pb, selection in itertools.product(
        encoding_types, crossover_types, mutation_types, crossover_probs, mutation_probs, selection_methods
    ):
        configs.append((encoding, crossover, mutation, cx_pb, mut_pb, selection))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, (encoding, crossover, mutation, cx_pb, mut_pb, selection) in enumerate(configs):
            print(f"Додаємо конфігурацію {i+1}/{total_configs} до черги виконання")
            future = executor.submit(
                run_generational_configuration,
                n_dims=n_dims,
                pop_size=pop_size,
                encoding_type=encoding,
                crossover_type=crossover,
                mutation_type=mutation,
                selection_method=selection,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                nr_runs=nr_runs,
                file_lock=file_lock
            )
            futures.append(future)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f"Завершено конфігурацію {i+1}/{total_configs}")
            future.result()
        
def run_steady_state_test(n_dims, pop_size, nr_runs):
    encoding_types = C.ENCODING_TYPES
    crossover_types = C.CROSSOVER_TYPES
    mutation_types = C.MUTATION_TYPES
    crossover_probs = C.CROSSOVER_PROBABILITIES[pop_size]
    mutation_probs = C.MUTATION_PROBABILITIES[pop_size]
    parent_selection_methods = C.SELECTION_PARENT_STEADY
    survivor_selection_methods = C.SELECTION_SURVIVOR_STEADY
    generation_gaps = C.GENERATION_GAPS

    total_configs = (len(encoding_types) * len(crossover_types) * len(mutation_types) * 
                     len(crossover_probs) * len(mutation_probs) * len(parent_selection_methods) * 
                     len(survivor_selection_methods) * len(generation_gaps))
    
    print(f"Запускаємо {total_configs} конфігурацій для стійкого типу")
    
    file_lock = multiprocessing.Manager().Lock()
    
    configs = []
    for encoding, crossover, mutation, cx_pb, mut_pb, parent_sel, surv_sel, gg in itertools.product(
        encoding_types, crossover_types, mutation_types, crossover_probs, mutation_probs, 
        parent_selection_methods, survivor_selection_methods, generation_gaps
    ):
        configs.append((encoding, crossover, mutation, cx_pb, mut_pb, parent_sel, surv_sel, gg))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, (encoding, crossover, mutation, cx_pb, mut_pb, parent_sel, surv_sel, gg) in enumerate(configs):
            print(f"Додаємо конфігурацію {i+1}/{total_configs} до черги виконання")
            future = executor.submit(
                run_steady_state_configuration,
                n_dims=n_dims,
                pop_size=pop_size,
                encoding_type=encoding,
                crossover_type=crossover,
                mutation_type=mutation,
                parent_selection=parent_sel,
                survivor_selection=surv_sel,
                generation_gap=gg,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                nr_runs=nr_runs,
                file_lock=file_lock
            )
            futures.append(future)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f"Завершено конфігурацію {i+1}/{total_configs}")
            try:
                future.result()
            except Exception as e:
                print(f"Помилка у виконанні конфігурації: {e}")

def main():
    parser = argparse.ArgumentParser(description="Автоматичне тестування генетичного алгоритму")
    
    parser.add_argument("--dims", type=int, required=True, choices=C.DIMENSIONS, help="Розмірність задачі")
    parser.add_argument("--pop_size", type=int, required=True, choices=C.POPULATION_SIZES, help="Розмір популяції")
    parser.add_argument("--repro_type", type=str, required=True, choices=C.REPRODUCTION_TYPES, help="Тип репродукції")
    parser.add_argument("--n_runs", type=int, default=C.NR, help="Кількість прогонів")
    
    args = parser.parse_args()
    
    print(f"Запуск тестування для dims={args.dims}, pop_size={args.pop_size}, repro_type={args.repro_type}")

    prepare_detail_file(args.repro_type, args.dims, args.pop_size, args.n_runs)
    prepare_summary_file(args.repro_type, args.dims, args.pop_size)

    start = time.time()
    if args.repro_type == "generational":
        run_generational_test(args.dims, args.pop_size, args.n_runs)
    else:
       run_steady_state_test(args.dims, args.pop_size, args.n_runs)
    
    print("Тестування завершено.")
    print(f"Час виконання: {(time.time() - start) / 60} хвилин")
if __name__ == "__main__":
    main() 