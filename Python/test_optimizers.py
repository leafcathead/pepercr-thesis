import unittest
import uuid

import numpy as np
import yaml

from Genetics import Chromosome

CONFIG_PATH = r'ConfigFiles/config.yaml'
GHC_FLAGS = []
CFG = None


class GeneticTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_segment(self):

        # Select a random set of active genes
        active_genes_size = 15  # Does not matter.
        active_genes = np.random.choice(GHC_FLAGS, size=active_genes_size, replace=False)

        # test a chromosome with 1 segment
        Chromosome.num_of_segments = 1
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 1)  # Test that correct number of segments was made
        self.assertEqual(len(self.deconstruct_seg_list(gene_segments)), len(GHC_FLAGS)) # Test that no flags were left out

        # test a chromosome with 3 segments
        Chromosome.num_of_segments = 3
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 3)  # Test that correct number of segments was made
        self.assertEqual(len(self.deconstruct_seg_list(gene_segments)), len(GHC_FLAGS)) # Test that no flags were left out

        # test a chromosome with 4 segments
        Chromosome.num_of_segments = 4
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 4)  # Test that correct number of segments was made
        self.assertEqual(len(self.deconstruct_seg_list(gene_segments)), len(GHC_FLAGS)) # Test that no flags were left out

        # test a chromosome with 30 segments
        Chromosome.num_of_segments = 30
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = c.get_gene_segments()

        self.assertEqual(len(gene_segments), 30)  # Test that correct number of segments was made
        self.assertEqual(len(self.deconstruct_seg_list(gene_segments)), len(GHC_FLAGS)) # Test that no flags were left out

        # test a chromosome with num of segments = to length of flags
        Chromosome.num_of_segments = len(GHC_FLAGS)
        c = Chromosome(active_genes, uuid.uuid4())


        self.assertEqual(len(gene_segments), len(GHC_FLAGS))  # Test that correct number of segments was made
        self.assertEqual(len(self.deconstruct_seg_list(gene_segments)), len(GHC_FLAGS)) # Test that no flags were left out

        # test a chromosome with 0 segments
        Chromosome.num_of_segments = 0
        c = Chromosome(active_genes, uuid.uuid4())
        gene_segments = []

        with self.assertRaises(Exception) as context:
            gene_segments = c.get_gene_segments()

        self.assertTrue("Did you configure the number of segments correctly in the configuration file?" in context.exception)



    def deconstruct_seg_list(self, seg_list):
        deconstructed_list = []
        for item in seg_list:
            for entry in item:
                deconstructed_list.append(entry)

        print(deconstructed_list)
        return deconstructed_list


if __name__ == '__main__':

    try:
        with open(CONFIG_PATH, "r") as cfg_file:
            CFG = yaml.safe_load(cfg_file)

        if CFG is None:
            raise IOError("CFG File is blank!")

        GHC_FLAGS = CFG["settings"]["flags"]

        # loads all unit tests from GeneticTests into a test suite
        genetic_optimizer_suite = unittest.TestLoader() \
            .loadTestsFromTestCase(GeneticTests)

        runner = unittest.TextTestRunner()
        runner.run(genetic_optimizer_suite)


    except IOError as e:
        print("Unable to open Configuration file")
        print(e)
