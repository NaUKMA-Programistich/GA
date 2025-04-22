import math
import random

import numpy as np
from deap import tools

import constants as C
import functions as F

def check_convergence_generational(population, avg_fitness_history, convergence_params):
    homogeneity_threshold = convergence_params.get('homogeneity_threshold', C.CONVERGENCE_HOMOGENEITY_THRESHOLD)
    stability_threshold = convergence_params.get('fitness_stability_threshold', C.CONVERGENCE_FITNESS_STABILITY_THRESHOLD)
    stability_window = convergence_params.get('fitness_stability_window', C.CONVERGENCE_FITNESS_STABILITY_WINDOW)
    
    is_homogeneous = F.check_homogeneity(population, homogeneity_threshold)
    
    is_stable = F.check_fitness_stability(avg_fitness_history, stability_window, stability_threshold)
    
    return is_homogeneous or is_stable

def check_convergence_steady_state(population, avg_fitness_history, convergence_params):
    homogeneity_threshold = convergence_params.get('homogeneity_threshold', C.CONVERGENCE_HOMOGENEITY_THRESHOLD)
    stability_threshold = convergence_params.get('fitness_stability_threshold', C.CONVERGENCE_FITNESS_STABILITY_THRESHOLD)
    stability_window = convergence_params.get('fitness_stability_window', C.CONVERGENCE_FITNESS_STABILITY_WINDOW)
    
    is_homogeneous = F.check_homogeneity(population, homogeneity_threshold)
    
    is_stable = F.check_fitness_stability(avg_fitness_history, stability_window, stability_threshold)
    
    return is_homogeneous or is_stable

def eaGenerational(
        population, 
        toolbox, 
        cxpb, 
        mutpb, 
        ngen, 
        stats=None,
        halloffame=None, 
        verbose=__debug__, 
        convergence_params=None
    ):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    nfe = 0

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    nfe += len(invalid_ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=nfe, **record)
    if verbose:
        print(logbook.stream)

    check_convergence = convergence_params is not None
    homogeneity_threshold = convergence_params.get('homogeneity_threshold', C.CONVERGENCE_HOMOGENEITY_THRESHOLD) if check_convergence else None
    stability_threshold = convergence_params.get('fitness_stability_threshold', C.CONVERGENCE_FITNESS_STABILITY_THRESHOLD) if check_convergence else None
    stability_window = convergence_params.get('fitness_stability_window', C.CONVERGENCE_FITNESS_STABILITY_WINDOW) if check_convergence else None
    avg_fitness_history = []
    if check_convergence:
         avg_fitness = np.mean([ind.fitness.values[0] for ind in population if ind.fitness.valid])
         avg_fitness_history.append(avg_fitness)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        nfe += len(invalid_ind)

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nfe, **record)
        if verbose:
            print(logbook.stream)

        converged = False
        if check_convergence:
            is_homogeneous = F.check_homogeneity(population, homogeneity_threshold)

            current_avg_fitness = np.mean([ind.fitness.values[0] for ind in population if ind.fitness.valid])
            avg_fitness_history.append(current_avg_fitness)
            is_stable = F.check_fitness_stability(avg_fitness_history, stability_window, stability_threshold)

            if is_homogeneous or is_stable:
                converged = True
                if verbose:
                    reason = "homogeneity" if is_homogeneous else "fitness stability"
                    print(f"Convergence reached at generation {gen} due to {reason}.")

        if converged:
            break

    return population, logbook, nfe


def eaSteadyState(
        population, 
        toolbox, 
        cxpb, 
        mutpb, 
        max_nfe, 
        gg,
        parent_selection_func, 
        survivor_selection_func,
        stats=None, 
        halloffame=None, 
        verbose=__debug__,
        convergence_params=None
    ):
    logbook = tools.Logbook()
    logbook.header = ['evals', 'nevals'] + (stats.fields if stats else [])

    nfe = 0
    iteration = 0

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    nfe += len(invalid_ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(evals=iteration, nevals=nfe, **record)
    if verbose:
        print(logbook.stream)

    check_convergence = convergence_params is not None
    homogeneity_threshold = convergence_params.get('homogeneity_threshold', C.CONVERGENCE_HOMOGENEITY_THRESHOLD) if check_convergence else None
    stability_threshold = convergence_params.get('fitness_stability_threshold', C.CONVERGENCE_FITNESS_STABILITY_THRESHOLD) if check_convergence else None
    base_stability_window = convergence_params.get('fitness_stability_window', C.CONVERGENCE_FITNESS_STABILITY_WINDOW) if check_convergence else 10
    stability_window = math.ceil(base_stability_window / gg) if check_convergence and gg > 0 else base_stability_window

    avg_fitness_history = []
    if check_convergence:
         avg_fitness = np.mean([ind.fitness.values[0] for ind in population if ind.fitness.valid])
         avg_fitness_history.append(avg_fitness)

    pop_size = len(population)
    num_breed = math.ceil(gg * pop_size)
    if num_breed < 2 and cxpb > 0:
        num_breed = 2

    while nfe < max_nfe:
        iteration += 1

        parents = parent_selection_func(population, num_breed)
        offspring = [toolbox.clone(ind) for ind in parents]

        for i in range(1, len(offspring), 2):
             if random.random() < cxpb:
                 offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                 del offspring[i-1].fitness.values
                 del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if not invalid_ind:
             continue

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        current_evals = len(invalid_ind)
        if nfe + current_evals > max_nfe:
             if verbose:
                 print(f"Stopping early: Evaluating {current_evals} individuals would exceed max_nfe ({max_nfe}). Current NFE: {nfe}")
             break
        nfe += current_evals


        population = survivor_selection_func(population, offspring)
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(evals=iteration, nevals=nfe, **record)
        if verbose:
            print(logbook.stream)

        converged = False
        if check_convergence:
            is_homogeneous = F.check_homogeneity(population, homogeneity_threshold)

            current_avg_fitness = np.mean([ind.fitness.values[0] for ind in population if ind.fitness.valid])
            avg_fitness_history.append(current_avg_fitness)
            is_stable = F.check_fitness_stability(avg_fitness_history, stability_window, stability_threshold)

            if is_homogeneous or is_stable:
                converged = True
                if verbose:
                    reason = "homogeneity" if is_homogeneous else "fitness stability"
                    print(f"Convergence reached at iteration {iteration} (NFE: {nfe}) due to {reason}.")

        if converged:
            break

    return population, logbook, nfe


def select_parents_elite(population, k):
    return tools.selBest(population, k)

def select_parents_rws(population, k):
    return tools.selRoulette(population, k)

def select_survivors_worst_comma(population, offspring):
    num_offspring = len(offspring)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    best_individuals = sorted_pop[num_offspring:]
    
    if not isinstance(best_individuals, list):
        best_individuals = list(best_individuals)
    if not isinstance(offspring, list):
        offspring = list(offspring)
        
    new_population = best_individuals + offspring
    
    return new_population

def select_survivors_rand_comma(population, offspring):
    pop_size = len(population)
    num_offspring = len(offspring)
    indices_to_replace = random.sample(range(pop_size), num_offspring)
    
    new_population = population.copy()
    for i, idx in enumerate(indices_to_replace):
        new_population[idx] = offspring[i]
    return new_population

def select_survivors_worst_plus(population, offspring):
    pop_size = len(population)
    combined = population + offspring
    return tools.selBest(combined, pop_size)

def select_survivors_rand_plus(population, offspring):
    pop_size = len(population)
    
    if not isinstance(offspring, list):
        offspring = list(offspring)
    combined = population + offspring
    
    return random.sample(combined, pop_size) 