import unittest
import constants

class TestConstants(unittest.TestCase):

    def test_function_constants(self):
        self.assertEqual(constants.A, 7.0)
        self.assertEqual(constants.X_MIN, -5.12)
        self.assertEqual(constants.X_MAX, 5.12)
        self.assertEqual(constants.PRECISION, 2)
        self.assertEqual(constants.OPTIMAL_X_PER_DIM, 0.0)

    def test_experiment_constants(self):
        self.assertEqual(constants.NR, 100)
        self.assertIsInstance(constants.NR, int)
        self.assertEqual(constants.DELTA, 0.01)
        self.assertIsInstance(constants.DELTA, float)
        self.assertEqual(constants.SIGMA, 0.01)
        self.assertIsInstance(constants.SIGMA, float)

    def test_ga_parameters(self):
        self.assertEqual(constants.MAX_GENERATIONS, 1_000_000)
        self.assertIsInstance(constants.MAX_GENERATIONS, int)

    def test_convergence_criteria(self):
        self.assertEqual(constants.CONVERGENCE_HOMOGENEITY_THRESHOLD, 0.99)
        self.assertEqual(constants.CONVERGENCE_FITNESS_STABILITY_THRESHOLD, 0.0001)
        self.assertEqual(constants.CONVERGENCE_FITNESS_STABILITY_WINDOW, 10)

    def test_parameter_ranges(self):
        self.assertIsInstance(constants.POPULATION_SIZES, list)
        self.assertIsInstance(constants.DIMENSIONS, list)
        self.assertIsInstance(constants.CROSSOVER_PROBABILITIES, dict)
        for pop_size in constants.POPULATION_SIZES:
            self.assertIn(pop_size, constants.CROSSOVER_PROBABILITIES)
            self.assertIsInstance(constants.CROSSOVER_PROBABILITIES[pop_size], list)
            self.assertEqual(constants.CROSSOVER_PROBABILITIES[pop_size], [0.0, 0.6, 0.8, 1.0])

        self.assertIsInstance(constants.MUTATION_PROBABILITIES, dict)
        for pop_size in constants.POPULATION_SIZES:
            self.assertIn(pop_size, constants.MUTATION_PROBABILITIES)
            self.assertIsInstance(constants.MUTATION_PROBABILITIES[pop_size], list)

        self.assertIsInstance(constants.GENERATION_GAPS, list)

    def test_component_types(self):
        self.assertIsInstance(constants.ENCODING_TYPES, list)
        self.assertIn("binary", constants.ENCODING_TYPES)
        self.assertIn("gray", constants.ENCODING_TYPES)
        self.assertIsInstance(constants.CROSSOVER_TYPES, list)
        self.assertIn("one_point", constants.CROSSOVER_TYPES)
        self.assertIn("uniform", constants.CROSSOVER_TYPES)
        self.assertIsInstance(constants.MUTATION_TYPES, list)
        self.assertIn("density", constants.MUTATION_TYPES)
        self.assertEqual(len(constants.MUTATION_TYPES), 1)
        self.assertIsInstance(constants.REPRODUCTION_TYPES, list)
        self.assertIn("generational", constants.REPRODUCTION_TYPES)
        self.assertIn("steady_state", constants.REPRODUCTION_TYPES)

    def test_selection_methods(self):
        self.assertIsInstance(constants.SELECTION_PARAMS_GENERATIONAL, dict)
        for pop_size in constants.POPULATION_SIZES:
            self.assertIn(pop_size, constants.SELECTION_PARAMS_GENERATIONAL)
            self.assertIsInstance(constants.SELECTION_PARAMS_GENERATIONAL[pop_size], list)
            self.assertGreater(len(constants.SELECTION_PARAMS_GENERATIONAL[pop_size]), 0)
            self.assertIsInstance(constants.SELECTION_PARAMS_GENERATIONAL[pop_size][0], str)
            if pop_size == 100:
                self.assertIn("ExpRankRWS_c=0.9801", constants.SELECTION_PARAMS_GENERATIONAL[pop_size])
                self.assertIn("LinRankSUS_beta=1.6", constants.SELECTION_PARAMS_GENERATIONAL[pop_size])

        self.assertIsInstance(constants.SELECTION_PARENT_STEADY, list)
        self.assertIsInstance(constants.SELECTION_SURVIVOR_STEADY, list)
        self.assertGreater(len(constants.SELECTION_PARENT_STEADY), 0)
        self.assertGreater(len(constants.SELECTION_SURVIVOR_STEADY), 0)

    def test_filenames(self):
        self.assertIsInstance(constants.DETAILED_STATS_FILENAME_TEMPLATE, str)
        self.assertIsInstance(constants.SUMMARY_STATS_FILENAME_TEMPLATE, str)
        self.assertIn("{repro_type}", constants.DETAILED_STATS_FILENAME_TEMPLATE)
        self.assertIn("{dim}", constants.DETAILED_STATS_FILENAME_TEMPLATE)
        self.assertIn("{pop_size}", constants.DETAILED_STATS_FILENAME_TEMPLATE)
        self.assertNotIn("{timestamp}", constants.DETAILED_STATS_FILENAME_TEMPLATE)

        self.assertIn("{repro_type}", constants.SUMMARY_STATS_FILENAME_TEMPLATE)
        self.assertIn("{dim}", constants.SUMMARY_STATS_FILENAME_TEMPLATE)
        self.assertIn("{pop_size}", constants.SUMMARY_STATS_FILENAME_TEMPLATE)
        self.assertNotIn("{timestamp}", constants.SUMMARY_STATS_FILENAME_TEMPLATE)

        self.assertTrue(constants.DETAILED_STATS_FILENAME_TEMPLATE.startswith("results/"))
        self.assertTrue(constants.SUMMARY_STATS_FILENAME_TEMPLATE.startswith("results/"))

    def test_encoding_parameters(self):
        self.assertEqual(constants.GENE_LENGTH_N1, 10)
        self.assertEqual(constants.BINOMIAL_P, 0.5)
        self.assertIsInstance(constants.GENE_LENGTH_N1, int)
        self.assertIsInstance(constants.BINOMIAL_P, float)

    def test_validation(self):
        constants.validate_constants()
        
        original_x_max = constants.X_MAX
        constants.X_MAX = constants.X_MIN - 1
        with self.assertRaises(ValueError):
            constants.validate_constants()
        constants.X_MAX = original_x_max
        
        original_precision = constants.PRECISION
        constants.PRECISION = 0
        with self.assertRaises(ValueError):
            constants.validate_constants()
        constants.PRECISION = original_precision
        
        original_crossover_probs = constants.CROSSOVER_PROBABILITIES[100].copy()
        constants.CROSSOVER_PROBABILITIES[100].append(1.1)
        with self.assertRaises(ValueError):
            constants.validate_constants()
        constants.CROSSOVER_PROBABILITIES[100] = original_crossover_probs
        
        original_mutation_probs = constants.MUTATION_PROBABILITIES[100].copy()
        constants.MUTATION_PROBABILITIES[100].append(-0.1)
        with self.assertRaises(ValueError):
            constants.validate_constants()
        constants.MUTATION_PROBABILITIES[100] = original_mutation_probs
        
        original_binomial_p = constants.BINOMIAL_P
        constants.BINOMIAL_P = 1.5
        with self.assertRaises(ValueError):
            constants.validate_constants()
        constants.BINOMIAL_P = original_binomial_p

if __name__ == '__main__':
    unittest.main() 