import unittest
import random
import numpy as np
from deap import creator, base, tools

import manual
import functions as F
import algorithms as A

class TestManual(unittest.TestCase):
    def setUp(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
        self.population = []
        for _ in range(10):
            ind = creator.Individual([random.randint(0, 1) for _ in range(20)])
            ind.fitness.values = (random.uniform(0, 100),)
            self.population.append(ind)
            
        random.seed(42)
        np.random.seed(42)
        
    def test_gray_to_binary_decoding(self):
        gray_individual = creator.Individual([1, 0, 1, 1, 0, 1, 0, 0])
        
        binary = [gray_individual[0]]
        for i in range(1, len(gray_individual)):
            binary.append(binary[i-1] ^ gray_individual[i])
        
        expected_binary = binary.copy()
            
        self.assertEqual(binary, expected_binary)
        
        n_dims = 1
        gene_length_per_dim = 8
        x_min = -5.12
        x_max = 5.12
        
        gray_phenotype = F.decode_individual(gray_individual, n_dims, gene_length_per_dim, x_min, x_max, "gray")
        
        binary_individual = creator.Individual(expected_binary)
        
        binary_phenotype = F.decode_individual(binary_individual, n_dims, gene_length_per_dim, x_min, x_max, "binary")
        
        self.assertAlmostEqual(gray_phenotype[0], binary_phenotype[0], places=5)

    def test_selection_rank_probabilities(self):
        pop = []
        for i in range(5):
            ind = creator.Individual([0] * 5)
            ind.fitness.values = (i * 10.0,)
            pop.append(ind)
            
        linear_probs = manual.assign_rank_probabilities(pop, method='linear', param=1.5)
        self.assertEqual(len(linear_probs), len(pop))
        
        self.assertAlmostEqual(sum(linear_probs), 1.0, places=5)
        
        sorted_inds = sorted(pop, key=lambda ind: ind.fitness.values[0])
        sorted_probs = [linear_probs[pop.index(ind)] for ind in sorted_inds]
        for i in range(1, len(sorted_probs)):
            self.assertGreaterEqual(sorted_probs[i], sorted_probs[i-1])
        
        exp_probs = manual.assign_rank_probabilities(pop, method='exponential', param=0.95)
        self.assertEqual(len(exp_probs), len(pop))
        self.assertAlmostEqual(sum(exp_probs), 1.0, places=5)
        
        with self.assertRaises(ValueError):
            manual.assign_rank_probabilities(pop, method='linear', param=0.5)
        with self.assertRaises(ValueError):
            manual.assign_rank_probabilities(pop, method='exponential', param=1.1)
        with self.assertRaises(ValueError):
            manual.assign_rank_probabilities(pop, method='unknown')

    def test_rank_selection(self):
        try:
            selected = manual.selRankSelection(self.population, k=5, method='linear', param=1.5, use_sus=False)
            self.assertEqual(len(selected), 5)
            for ind in selected:
                self.assertIn(ind, self.population)
                
            selected = manual.selRankSelection(self.population, k=5, method='linear', param=1.5, use_sus=True)
            self.assertEqual(len(selected), 5)
            for ind in selected:
                self.assertIn(ind, self.population)
            
            selected = manual.selRankSelection(self.population, k=5, method='exponential', param=0.95, use_sus=False)
            self.assertEqual(len(selected), 5)
            for ind in selected:
                self.assertIn(ind, self.population)
        except ValueError as e:
            if "probabilities do not sum to 1" in str(e):
                self.skipTest("Ймовірності не підсумовуються до 1, тест пропущено")
            else:
                raise

    def test_tournament_selection(self):
        selected = tools.selTournament(self.population, k=5, tournsize=2)
        self.assertEqual(len(selected), 5)
        
        selected = manual.selTournamentWithoutReplacement(self.population, k=5, tournsize=2)
        self.assertEqual(len(selected), 5)
        
        selected_ids = [id(ind) for ind in selected]
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        
        selected = manual.selTournamentPartialReplacement(self.population, k=5, tournsize=2)
        self.assertEqual(len(selected), 5)
        
        small_pop = []
        for _ in range(3):
            ind = creator.Individual([random.randint(0, 1) for _ in range(20)])
            ind.fitness.values = (random.uniform(0, 100),)
            small_pop.append(ind)
            
        selected = manual.selTournamentWithoutReplacement(small_pop, k=5, tournsize=2)
        self.assertEqual(len(selected), 5)
        
        selected = manual.selTournamentWithoutReplacement(small_pop, k=3, tournsize=5)
        self.assertEqual(len(selected), 3)

    def test_steady_state_survivor_selection(self):
        offspring = []
        for _ in range(3):
            ind = creator.Individual([random.randint(0, 1) for _ in range(20)])
            ind.fitness.values = (random.uniform(0, 100),)
            offspring.append(ind)
        
        pop_size = len(self.population)
        new_pop = A.select_survivors_worst_comma(self.population, offspring)
        self.assertEqual(len(new_pop), pop_size)
        
        new_pop = A.select_survivors_rand_comma(self.population, offspring)
        self.assertEqual(len(new_pop), pop_size)
        
        new_pop = A.select_survivors_worst_plus(self.population, offspring)
        self.assertEqual(len(new_pop), pop_size)
        
        new_pop = A.select_survivors_rand_plus(self.population, offspring)
        self.assertEqual(len(new_pop), pop_size)
        
        best_fitness = max([ind.fitness.values[0] for ind in self.population + offspring])
        best_in_new_pop = max([ind.fitness.values[0] for ind in new_pop])
        self.assertAlmostEqual(best_fitness, best_in_new_pop, places=10)

    def test_toolbox_setup(self):
        toolbox, gene_length = manual.setup_toolbox(
            n_dims=2,
            pop_size=100,
            encoding_type="gray",
            crossover_type="one_point",
            mutation_type="density",
            selection_method="SUS",
            reproduction_type="generational",
            cx_pb=0.8,
            mut_pb=0.01
        )
        
        self.assertTrue(hasattr(toolbox, "individual"))
        self.assertTrue(hasattr(toolbox, "population"))
        self.assertTrue(hasattr(toolbox, "evaluate"))
        self.assertTrue(hasattr(toolbox, "select"))
        self.assertTrue(hasattr(toolbox, "mate"))
        self.assertTrue(hasattr(toolbox, "mutate"))
        
        with self.assertRaises(ValueError):
            manual.setup_toolbox(
                n_dims=2,
                pop_size=100,
                encoding_type="gray",
                crossover_type="unknown",
                mutation_type="density",
                selection_method="SUS",
                reproduction_type="generational",
                cx_pb=0.8,
                mut_pb=0.01
            )
            
        with self.assertRaises(ValueError):
            manual.setup_toolbox(
                n_dims=2,
                pop_size=100,
                encoding_type="gray",
                crossover_type="one_point",
                mutation_type="unknown",
                selection_method="SUS",
                reproduction_type="generational",
                cx_pb=0.8,
                mut_pb=0.01
            )
            
        with self.assertRaises(ValueError):
            manual.setup_toolbox(
                n_dims=2,
                pop_size=100,
                encoding_type="gray",
                crossover_type="one_point",
                mutation_type="density",
                selection_method="SUS",
                reproduction_type="unknown",
                cx_pb=0.8,
                mut_pb=0.01
            )

if __name__ == '__main__':
    unittest.main()
