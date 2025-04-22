import unittest
import random
import numpy as np
from deap import base, creator, tools

import algorithms as A

def evaluate_dummy(individual):
    return sum(individual),

class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        self.toolbox = base.Toolbox()

        self.gene_length = 10
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, self.gene_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", evaluate_dummy)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.pop_size = 20
        self.cxpb = 0.6
        self.mutpb = 0.2
        self.ngen = 5
        self.max_nfe_steady = 100
        self.gg = 0.5

        self.test_pop = self.toolbox.population(n=self.pop_size)
        fitnesses = map(self.toolbox.evaluate, self.test_pop)
        for ind, fit in zip(self.test_pop, fitnesses):
            ind.fitness.values = fit

        self.sorted_test_pop = sorted(self.test_pop, key=lambda ind: ind.fitness.values[0], reverse=True)

    def test_eaGenerational_runs(self):
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        final_pop, logbook, nfe = A.eaGenerational(
            pop, self.toolbox, self.cxpb, self.mutpb, self.ngen,
            stats=stats, halloffame=hof, verbose=False
        )

        self.assertIsInstance(final_pop, list)
        self.assertEqual(len(final_pop), self.pop_size)
        self.assertIsInstance(logbook, tools.Logbook)
        self.assertEqual(len(logbook), self.ngen + 1)
        self.assertIsInstance(nfe, int)
        self.assertGreater(nfe, self.pop_size)
        self.assertTrue(all(ind.fitness.valid for ind in final_pop))
        self.assertEqual(len(hof), 1)

    def test_eaGenerational_convergence_check(self):
        pop = self.toolbox.population(n=self.pop_size)
        convergence_params = {
            'homogeneity_threshold': 0.99,
            'fitness_stability_threshold': 0.0001,
            'fitness_stability_window': 3
        }
        final_pop, logbook, nfe = A.eaGenerational(
            pop, self.toolbox, self.cxpb, self.mutpb, 2,
            verbose=False, convergence_params=convergence_params
        )
        self.assertIsInstance(final_pop, list)
        self.assertEqual(len(final_pop), self.pop_size)
        self.assertLessEqual(len(logbook), 2 + 1)

    def test_eaSteadyState_runs(self):
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        parent_selector = A.select_parents_elite
        survivor_selector = A.select_survivors_worst_plus

        final_pop, logbook, nfe = A.eaSteadyState(
            pop, self.toolbox, self.cxpb, self.mutpb, self.max_nfe_steady, self.gg,
            parent_selection_func=parent_selector,
            survivor_selection_func=survivor_selector,
            stats=stats, halloffame=hof, verbose=False
        )

        self.assertIsInstance(final_pop, list)
        self.assertEqual(len(final_pop), self.pop_size)
        self.assertIsInstance(logbook, tools.Logbook)
        self.assertGreater(len(logbook), 1)
        self.assertIsInstance(nfe, int)
        self.assertLessEqual(nfe, self.max_nfe_steady)
        self.assertTrue(all(ind.fitness.valid for ind in final_pop))
        self.assertEqual(len(hof), 1)

    def test_eaSteadyState_convergence_check(self):
        pop = self.toolbox.population(n=self.pop_size)
        convergence_params = {
            'homogeneity_threshold': 0.99,
            'fitness_stability_threshold': 0.0001,
            'fitness_stability_window': 3
        }
        parent_selector = A.select_parents_elite
        survivor_selector = A.select_survivors_worst_plus

        final_pop, logbook, nfe = A.eaSteadyState(
            pop, self.toolbox, self.cxpb, self.mutpb, 50, self.gg,
            parent_selection_func=parent_selector,
            survivor_selection_func=survivor_selector,
            verbose=False, convergence_params=convergence_params
        )
        self.assertIsInstance(final_pop, list)
        self.assertEqual(len(final_pop), self.pop_size)
        self.assertLessEqual(nfe, 50)
        
    def test_eaSteadyState_with_rand_comma(self):
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        parent_selector = A.select_parents_elite
        survivor_selector = A.select_survivors_rand_comma

        final_pop, logbook, nfe = A.eaSteadyState(
            pop, self.toolbox, self.cxpb, self.mutpb, self.max_nfe_steady, self.gg,
            parent_selection_func=parent_selector,
            survivor_selection_func=survivor_selector,
            stats=stats, halloffame=hof, verbose=False,
            convergence_params={
                'homogeneity_threshold': 0.99,
                'fitness_stability_threshold': 0.0001,
                'fitness_stability_window': 3
            }
        )

        self.assertIsInstance(final_pop, list)
        self.assertEqual(len(final_pop), self.pop_size)
        self.assertIsInstance(logbook, tools.Logbook)
        self.assertGreater(len(logbook), 1)
        self.assertIsInstance(nfe, int)
        self.assertLessEqual(nfe, self.max_nfe_steady)
        self.assertTrue(all(ind.fitness.valid for ind in final_pop))
        self.assertEqual(len(hof), 1)

    def test_select_parents_elite(self):
        k = 5
        selected = A.select_parents_elite(self.test_pop, k)
        self.assertEqual(len(selected), k)
        self.test_pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        expected_best_fitness = [ind.fitness.values[0] for ind in self.test_pop[:k]]
        selected_fitness = [ind.fitness.values[0] for ind in selected]
        self.assertListEqual(sorted(selected_fitness, reverse=True), sorted(expected_best_fitness, reverse=True))

    def test_select_parents_rws(self):
        k = 5
        selected = A.select_parents_rws(self.test_pop, k)
        self.assertEqual(len(selected), k)
        self.assertTrue(all(isinstance(ind, creator.Individual) for ind in selected))

    def test_select_survivors_worst_comma(self):
        num_offspring = int(self.pop_size * self.gg)
        offspring = self.toolbox.population(n=num_offspring)
        for i, ind in enumerate(offspring):
            ind.fitness.values = (1000 + i,)
        original_pop_copy = [self.toolbox.clone(ind) for ind in self.test_pop]
        
        original_sorted = sorted(original_pop_copy, key=lambda ind: ind.fitness.values[0])
        worst_original = original_sorted[:num_offspring]

        new_pop = A.select_survivors_worst_comma(original_pop_copy, offspring)
        self.assertEqual(len(new_pop), self.pop_size)
        
        offspring_fitnesses = {ind.fitness.values[0] for ind in offspring}
        new_pop_fitnesses = {ind.fitness.values[0] for ind in new_pop}
        self.assertTrue(offspring_fitnesses.issubset(new_pop_fitnesses))

        for ind in worst_original:
            self.assertFalse(any(ind is pop_ind for pop_ind in new_pop),
                            f"Індивід з фітнесом {ind.fitness.values[0]} не повинен бути в новій популяції")

    def test_select_survivors_rand_comma(self):
        num_offspring = int(self.pop_size * self.gg)
        offspring = self.toolbox.population(n=num_offspring)
        for ind in offspring:
            ind.fitness.values = evaluate_dummy(ind)

        original_pop_copy = [self.toolbox.clone(ind) for ind in self.test_pop]
        new_pop = A.select_survivors_rand_comma(original_pop_copy, offspring)

        self.assertEqual(len(new_pop), self.pop_size)

        replaced_count = 0
        for ind_new in new_pop:
            is_offspring = False
            for ind_off in offspring:
                if ind_new is ind_off:
                    is_offspring = True
                    break
            if is_offspring:
                replaced_count += 1

        self.assertEqual(replaced_count, num_offspring)
        
    def test_select_survivors_rand_comma_equal_size(self):
        offspring = self.toolbox.population(n=self.pop_size)
        for ind in offspring:
            ind.fitness.values = evaluate_dummy(ind)

        original_pop_copy = [self.toolbox.clone(ind) for ind in self.test_pop]
        new_pop = A.select_survivors_rand_comma(original_pop_copy, offspring)

        self.assertEqual(len(new_pop), self.pop_size)
        
        offspring_ids = {id(ind) for ind in offspring}
        new_pop_ids = {id(ind) for ind in new_pop}
        self.assertEqual(offspring_ids, new_pop_ids)
        
    def test_select_survivors_rand_comma_small_offspring(self):
        num_offspring = 1
        offspring = self.toolbox.population(n=num_offspring)
        for ind in offspring:
            ind.fitness.values = evaluate_dummy(ind)

        original_pop_copy = [self.toolbox.clone(ind) for ind in self.test_pop]
        new_pop = A.select_survivors_rand_comma(original_pop_copy, offspring)

        self.assertEqual(len(new_pop), self.pop_size)
        
        replaced_count = 0
        for ind_new in new_pop:
            if ind_new is offspring[0]:
                replaced_count += 1
                
        self.assertEqual(replaced_count, 1)

    def test_select_survivors_worst_plus(self):
        num_offspring = int(self.pop_size * self.gg)
        offspring = self.toolbox.population(n=num_offspring)
        for ind in offspring:
            ind.fitness.values = evaluate_dummy(ind)

        combined_pop = self.test_pop + offspring
        combined_pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        expected_best_fitness = [ind.fitness.values[0] for ind in combined_pop[:self.pop_size]]

        new_pop = A.select_survivors_worst_plus(self.test_pop, offspring)
        self.assertEqual(len(new_pop), self.pop_size)

        new_pop_fitness = [ind.fitness.values[0] for ind in new_pop]
        self.assertListEqual(sorted(new_pop_fitness, reverse=True), sorted(expected_best_fitness, reverse=True))

    def test_select_survivors_rand_plus(self):
        num_offspring = int(self.pop_size * self.gg)
        offspring = self.toolbox.population(n=num_offspring)
        for ind in offspring:
            ind.fitness.values = evaluate_dummy(ind)

        new_pop = A.select_survivors_rand_plus(self.test_pop, offspring)
        self.assertEqual(len(new_pop), self.pop_size)
        original_ids = {id(ind) for ind in self.test_pop}
        offspring_ids = {id(ind) for ind in offspring}
        combined_ids = original_ids.union(offspring_ids)
        new_pop_ids = {id(ind) for ind in new_pop}
        self.assertTrue(new_pop_ids.issubset(combined_ids))

    def test_convergence_check_generational(self):
        homogeneous_pop = [[1, 0, 1, 0, 1]] * 10
        convergence_params = {
            'homogeneity_threshold': 0.99,
            'fitness_stability_threshold': 0.0001,
            'fitness_stability_window': 3
        }
        fitness_history = [42.0] * 5
        self.assertTrue(A.check_convergence_generational(homogeneous_pop, fitness_history, convergence_params))

        diverse_pop = [[1, 0, 1, 0, 1]] * 9 + [[0, 1, 0, 1, 0]]
        stable_fitness = [42.0] * 15
        self.assertTrue(A.check_convergence_generational(diverse_pop, stable_fitness, convergence_params))

        unstable_fitness = [40.0, 41.0, 42.0, 43.0, 44.0]
        self.assertFalse(A.check_convergence_generational(diverse_pop, unstable_fitness, convergence_params))

    def test_convergence_check_steady_state(self):
        homogeneous_pop = [[1, 0, 1, 0, 1]] * 10
        convergence_params = {
            'homogeneity_threshold': 0.99,
            'fitness_stability_threshold': 0.0001,
            'fitness_stability_window': 3
        }
        fitness_history = [42.0] * 5
        self.assertTrue(A.check_convergence_steady_state(homogeneous_pop, fitness_history, convergence_params))

        diverse_pop = [[1, 0, 1, 0, 1]] * 9 + [[0, 1, 0, 1, 0]]
        stable_fitness = [42.0] * 15
        self.assertTrue(A.check_convergence_steady_state(diverse_pop, stable_fitness, convergence_params))

        unstable_fitness = [40.0, 41.0, 42.0, 43.0, 44.0]
        self.assertFalse(A.check_convergence_steady_state(diverse_pop, unstable_fitness, convergence_params))

if __name__ == '__main__':
    unittest.main() 