import random
import unittest
import math
import numpy as np
from deap import creator, tools

import functions as F
import constants as C

def init_population_with_seed(toolbox, seed, pop_size):
    random.seed(seed)
    np.random.seed(seed)
    return toolbox.population(n=pop_size)

class TestFunctions(unittest.TestCase):

    def test_calculate_gene_length(self):
        self.assertEqual(F.calculate_gene_length(-5.12, 5.12, 2), 11)
        self.assertEqual(F.calculate_gene_length(0, 1, 0), 1)
        self.assertEqual(F.calculate_gene_length(0, 1, 1), 4)
        self.assertEqual(F.calculate_gene_length(-1, 1, 1), 5)

    def test_get_optimal_fitness(self):
        self.assertAlmostEqual(F.get_optimal_fitness(1, C.A, C.OPTIMAL_X_PER_DIM), 49.0)
        self.assertAlmostEqual(F.get_optimal_fitness(2, C.A, C.OPTIMAL_X_PER_DIM), 98.0)
        self.assertAlmostEqual(F.get_optimal_fitness(3, C.A, C.OPTIMAL_X_PER_DIM), 147.0)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(F.euclidean_distance([0], [3]), 3.0)
        self.assertAlmostEqual(F.euclidean_distance([0, 0], [3, 4]), 5.0)
        self.assertAlmostEqual(F.euclidean_distance([1, 2, 3], [1, 2, 3]), 0.0)
        with self.assertRaises(ValueError):
            F.euclidean_distance([1, 2], [1, 2, 3])

    def test_decode_binary_segment(self):
        gene_len = 10
        x_min, x_max = -5.12, 5.12
        
        self.assertAlmostEqual(F.decode_binary_segment([0]*gene_len, x_min, x_max, gene_len), x_min)

        max_val_expected = x_max - 10**(-C.PRECISION - 1)
        self.assertAlmostEqual(F.decode_binary_segment([1]*gene_len, x_min, x_max, gene_len), max_val_expected)
        
        segment = [1] + [0]*9
        expected = x_min + (512 / (2**gene_len - 1)) * (x_max - x_min)
        self.assertAlmostEqual(F.decode_binary_segment(segment, x_min, x_max, gene_len), expected)

    def test_decode_gray_segment(self):
        gene_len = 10
        x_min, x_max = -5.12, 5.12
        
        self.assertAlmostEqual(F.decode_gray_segment([0]*gene_len, x_min, x_max, gene_len), x_min)
        
        gray_max = [1] + [0]*(gene_len-1)
        decoded_val = F.decode_gray_segment(gray_max, x_min, x_max, gene_len)
        self.assertGreater(decoded_val, x_min)
        
        gray_code = [1, 0, 1, 0]
        binary_code = [1, 1, 0, 0]
        
        converted_binary = [gray_code[0]]
        for i in range(1, len(gray_code)):
            converted_binary.append(converted_binary[i-1] ^ gray_code[i])
            
        self.assertEqual(converted_binary, binary_code)

    def test_decode_individual(self):
        n_dims = 2
        gene_length_per_dim = 10
        x_min, x_max = -5.12, 5.12
        
        binary_ind = [0]*10 + [1]*10
        binary_phenotype = F.decode_individual(binary_ind, n_dims, gene_length_per_dim, x_min, x_max, "binary")
        self.assertEqual(len(binary_phenotype), n_dims)
        self.assertAlmostEqual(binary_phenotype[0], x_min)
        self.assertAlmostEqual(binary_phenotype[1], x_max - 10**(-C.PRECISION - 1))
        
        gray_ind = [0]*10 + [1] + [0]*9
        gray_phenotype = F.decode_individual(gray_ind, n_dims, gene_length_per_dim, x_min, x_max, "gray")
        self.assertEqual(len(gray_phenotype), n_dims)
        
        with self.assertRaises(ValueError):
            F.decode_individual(binary_ind, n_dims, gene_length_per_dim, x_min, x_max, "unknown")
            
        with self.assertRaises(ValueError):
            F.decode_individual(binary_ind[:-1], n_dims, gene_length_per_dim, x_min, x_max, "binary")

    def test_evaluate_target_function(self):
        a = C.A
        
        optimal_phenotype_1d = [0.0]
        optimal_phenotype_2d = [0.0, 0.0]
        
        expected_1d = 1 * abs(10 * math.cos(2 * math.pi * a) - a**2) + 10 * math.cos(0) - 0
        expected_2d = 2 * abs(10 * math.cos(2 * math.pi * a) - a**2) + 10 * math.cos(0) - 0 + 10 * math.cos(0) - 0
        
        self.assertAlmostEqual(F.evaluate_target_function(optimal_phenotype_1d, a), expected_1d)
        self.assertAlmostEqual(F.evaluate_target_function(optimal_phenotype_2d, a), expected_2d)
        
        with self.assertRaises(ValueError):
            F.evaluate_target_function([], a)

    def test_evaluate(self):
        n_dims = 1
        gene_length_per_dim = F.calculate_gene_length(C.X_MIN, C.X_MAX, C.PRECISION)
        x_min, x_max = C.X_MIN, C.X_MAX
        a = C.A
        
        ind_min = creator.Individual([0] * gene_length_per_dim)
        fitness_min = F.evaluate(ind_min, n_dims, gene_length_per_dim, "binary", a, x_min, x_max)
        
        ind_max = creator.Individual([1] * gene_length_per_dim)
        fitness_max = F.evaluate(ind_max, n_dims, gene_length_per_dim, "binary", a, x_min, x_max)
        
        self.assertIsInstance(fitness_min, tuple)
        self.assertEqual(len(fitness_min), 1)
        self.assertIsInstance(fitness_max, tuple)
        self.assertEqual(len(fitness_max), 1)
        
        fitness_gray = F.evaluate(ind_max, n_dims, gene_length_per_dim, "gray", a, x_min, x_max)
        self.assertIsInstance(fitness_gray, tuple)
        self.assertEqual(len(fitness_gray), 1)

    def test_check_homogeneity(self):
        homogeneous_pop = [[1, 0, 1, 0, 1]] * 10
        self.assertTrue(F.check_homogeneity(homogeneous_pop, 0.99))
        
        diverse_pop = [[1, 0, 1, 0, 1]] * 9 + [[0, 1, 0, 1, 0]]
        self.assertFalse(F.check_homogeneity(diverse_pop, 0.99))
        
        self.assertFalse(F.check_homogeneity([], 0.99))
        self.assertTrue(F.check_homogeneity([[1, 0, 1]], 0.99))
        
        self.assertTrue(F.check_homogeneity(diverse_pop, 0.8))
        
        mixed_pop = [[1, 0, 1, 0, 1]] * 9 + [[0, 0, 1, 0, 1]]
        self.assertFalse(F.check_homogeneity(mixed_pop, 0.99))
        self.assertTrue(F.check_homogeneity(mixed_pop, 0.9))
        
        fully_diverse = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 0, 1]]
        self.assertFalse(F.check_homogeneity(fully_diverse, 0.8))

    def test_check_fitness_stability(self):
        stable_fitness = [42.0] * 15
        self.assertTrue(F.check_fitness_stability(stable_fitness, 10, 0.0001))
        
        unstable_fitness = [42.0, 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7, 42.8, 42.9, 43.0]
        self.assertFalse(F.check_fitness_stability(unstable_fitness, 10, 0.0001))
        
        insufficient_data = [42.0, 42.0, 42.0]
        self.assertFalse(F.check_fitness_stability(insufficient_data, 10, 0.0001))
        
        self.assertTrue(F.check_fitness_stability(unstable_fitness, 10, 0.2))

        borderline_stable = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0001]
        self.assertTrue(F.check_fitness_stability(borderline_stable, 10, 0.0001))
        
        slightly_unstable = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.00011]
        self.assertFalse(F.check_fitness_stability(slightly_unstable, 10, 0.0001))
        
        window_test = [50.0, 50.1, 50.2, 50.3, 50.4]
        self.assertFalse(F.check_fitness_stability(window_test, 4, 0.05))
        self.assertTrue(F.check_fitness_stability(window_test, 2, 0.15))

    def test_check_success(self):
        n_dims = 1
        gene_len = F.calculate_gene_length(C.X_MIN, C.X_MAX, C.PRECISION)
        x_min, x_max = C.X_MIN, C.X_MAX
        a = C.A
        optimal_fitness = F.get_optimal_fitness(n_dims, a, C.OPTIMAL_X_PER_DIM)
        optimal_x = C.OPTIMAL_X_PER_DIM
        delta = C.DELTA
        sigma = C.SIGMA

        ind_success = creator.Individual([1] + [0]*(gene_len-1))
        ind_success.fitness.values = F.evaluate(ind_success, n_dims, gene_len, "binary", a, x_min, x_max)
        self.assertGreaterEqual(ind_success.fitness.values[0], (1 - delta) * optimal_fitness)

        decoded_val = F.decode_individual(ind_success, n_dims, gene_len, x_min, x_max, "binary")[0]
        self.assertLessEqual(abs(decoded_val - optimal_x), sigma)

        is_succ, pa, da = F.check_success(ind_success, n_dims, gene_len, "binary", a, x_min, x_max, optimal_fitness, optimal_x, delta, sigma)
        self.assertTrue(is_succ)
        self.assertAlmostEqual(pa, ind_success.fitness.values[0] / optimal_fitness)
        self.assertAlmostEqual(da, abs(decoded_val - optimal_x))

        ind_far = creator.Individual([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        decoded_far_val = F.decode_individual(ind_far, n_dims, gene_len, x_min, x_max, "binary")[0]
        self.assertGreater(abs(decoded_far_val - optimal_x), sigma)

        ind_far.fitness.values = (optimal_fitness * (1 - delta / 2),)
        is_succ, pa, da = F.check_success(ind_far, n_dims, gene_len, "binary", a, x_min, x_max, optimal_fitness, optimal_x, delta, sigma)
        self.assertFalse(is_succ)
        self.assertGreater(da, sigma)
        self.assertAlmostEqual(pa, ind_far.fitness.values[0] / optimal_fitness)

        ind_low_fit = creator.Individual([1] + [0]*(gene_len-1))
        decoded_close_val = F.decode_individual(ind_low_fit, n_dims, gene_len, x_min, x_max, "binary")[0]
        self.assertLessEqual(abs(decoded_close_val - optimal_x), sigma)

        ind_low_fit.fitness.values = (optimal_fitness * (1 - 2*delta),)
        is_succ, pa, da = F.check_success(ind_low_fit, n_dims, gene_len, "binary", a, x_min, x_max, optimal_fitness, optimal_x, delta, sigma)
        self.assertFalse(is_succ)
        self.assertLess(ind_low_fit.fitness.values[0], (1 - delta) * optimal_fitness)
        self.assertLessEqual(da, sigma)

        ind_bad_all = creator.Individual([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        ind_bad_all.fitness.values = (optimal_fitness * (1 - 2*delta),)
        is_succ, pa, da = F.check_success(ind_bad_all, n_dims, gene_len, "binary", a, x_min, x_max, optimal_fitness, optimal_x, delta, sigma)
        self.assertFalse(is_succ)
        decoded_bad_val = F.decode_individual(ind_bad_all, n_dims, gene_len, x_min, x_max, "binary")[0]
        self.assertGreater(abs(decoded_bad_val - optimal_x), sigma)
        self.assertLess(ind_bad_all.fitness.values[0], (1 - delta) * optimal_fitness)

        ind_invalid_fit = creator.Individual([1] + [0]*(gene_len-1))
        is_succ, pa, da = F.check_success(ind_invalid_fit, n_dims, gene_len, "binary", a, x_min, x_max, optimal_fitness, optimal_x, delta, sigma)
        self.assertFalse(is_succ)
        self.assertEqual(pa, 0.0)
        self.assertEqual(da, float('inf'))

    def test_init_individual(self):
        size = 50
        p = C.BINOMIAL_P

        np.random.seed(42)
        ind = F.init_individual(creator.Individual, size, p)

        self.assertIsInstance(ind, creator.Individual)
        self.assertEqual(len(ind), size)
        self.assertTrue(all(bit in [0, 1] for bit in ind))

        num_ones = sum(ind)
        expected_ones = size * p
        std_dev = math.sqrt(size * p * (1 - p))
        self.assertLessEqual(abs(num_ones - expected_ones), 4 * std_dev)

    def test_init_population_with_seed(self):
        from deap import base
        
        toolbox = base.Toolbox()
        toolbox.register("individual", F.init_individual, creator.Individual, 10, C.BINOMIAL_P)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        seed = 42
        pop_size = 5
        
        pop1 = init_population_with_seed(toolbox, seed, pop_size)
        pop2 = init_population_with_seed(toolbox, seed, pop_size)
        
        for ind1, ind2 in zip(pop1, pop2):
            self.assertEqual(ind1, ind2)
        
        pop3 = init_population_with_seed(toolbox, seed+1, pop_size)
        
        at_least_one_different = False
        for ind1, ind3 in zip(pop1, pop3):
            if ind1 != ind3:
                at_least_one_different = True
                break
        self.assertTrue(at_least_one_different)

if __name__ == '__main__':
    unittest.main()