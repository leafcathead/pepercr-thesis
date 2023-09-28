import random
import unittest
import uuid

import numpy as np
import yaml

from Genetics import Chromosome

CONFIG_PATH = r'ConfigFiles/config.yaml'


def deconstruct_seg_list(seg_list):
    deconstructed_list = []
    for item in seg_list:
        for entry in item:
            deconstructed_list.append(entry)

    return deconstructed_list




class GeneticTests(unittest.TestCase):
    GHC_FLAGS = None
    CFG = None

    def create_random_chromosome(self):
        return Chromosome(np.random.choice(self.GHC_FLAGS, size=random.randint(0, len(self.GHC_FLAGS) - 1), replace=False), uuid.uuid4())

    @classmethod
    def setUpClass(cls):
        try:
            with open(CONFIG_PATH, "r") as cfg_file:
                cls.CFG = yaml.safe_load(cfg_file)

            if cls.CFG is None:
                raise IOError("CFG File is blank!")

            cls.GHC_FLAGS = cls.CFG["settings"]["flags"]
            Chromosome.genes = cls.GHC_FLAGS

        except IOError as e:
            print("Unable to open Configuration file")
            print(e)

    def test_segment(self):

        # Select a random set of active genes
        active_genes_size = 15  # Does not matter.
        active_genes = np.random.choice(self.GHC_FLAGS, size=active_genes_size, replace=False)

        # test a chromosome with 1 segment
        Chromosome.num_of_segments = 1
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 1)  # Test that correct number of segments was made
        self.assertFalse(set(deconstruct_seg_list(gene_segments)) - (
            set(["-O0"] + self.GHC_FLAGS)))  # Test that no flags were left out

        # test a chromosome with 3 segments
        Chromosome.num_of_segments = 3
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 3)  # Test that correct number of segments was made
        self.assertFalse(set(deconstruct_seg_list(gene_segments)) - (
            set(["-O0"] + self.GHC_FLAGS)))  # Test that no flags were left out

        # test a chromosome with 4 segments
        Chromosome.num_of_segments = 4
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 4)  # Test that correct number of segments was made
        self.assertFalse(set(deconstruct_seg_list(gene_segments)) - (
            set(["-O0"] + self.GHC_FLAGS)))  # Test that no flags were left out

        # test a chromosome with 30 segments
        Chromosome.num_of_segments = 30
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 20)  # Test that correct number of segments was made
        self.assertFalse(set(deconstruct_seg_list(gene_segments)) - (
            set(["-O0"] + self.GHC_FLAGS)))  # Test that no flags were left out

        # test a chromosome with num of segments = to length of flags
        Chromosome.num_of_segments = len(self.GHC_FLAGS)
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), len(self.GHC_FLAGS) + 1)  # Test that correct number of segments was made
        self.assertFalse(set(deconstruct_seg_list(gene_segments)) - (
            set(["-O0"] + self.GHC_FLAGS)))  # Test that no flags were left out

        # test a chromosome with 0 segments
        Chromosome.num_of_segments = 0
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = []

        with self.assertRaises(Exception) as context:
            gene_segments = c.get_gene_segments()

        self.assertEqual(
            "Did you configure the number of segments correctly in the configuration file?", context.exception.args[0])

    def test_crossover(self):

        # Set up test chromosomes

        c0 = Chromosome(self.GHC_FLAGS[:20], 0)
        c1 = Chromosome(self.GHC_FLAGS[20:40], 1)
        c2 = Chromosome(self.GHC_FLAGS[:40], 2)
        c3 = Chromosome(self.GHC_FLAGS[:10] + self.GHC_FLAGS[20:30], 3)
        c4 = Chromosome(self.GHC_FLAGS[40:50], 4)
        c5 = Chromosome(self.GHC_FLAGS[5:10] + self.GHC_FLAGS[15:20] + self.GHC_FLAGS[25:30] + self.GHC_FLAGS[35:40], 5)

        c0.fitness = 0.0
        c1.fitness = 0.1
        c2.fitness = 0.2
        c3.fitness = 0.3
        c4.fitness = 0.4
        c5.fitness = 0.5

        pass






if __name__ == '__main__':
    # loads all unit tests from GeneticTests into a test suite
    genetic_optimizer_suite = unittest.TestLoader() \
        .loadTestsFromTestCase(GeneticTests)

    runner = unittest.TextTestRunner()
    runner.run(genetic_optimizer_suite)
