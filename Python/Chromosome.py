# Used to help abstract the genetic algorithm

class Chromosome:
    genes = []

    def __init__(self, active_genes):
        self.sequence = dict()  # 00100100
        # 1 - flag is enabled in chromosome
        # 0 - flag is disabled in chromosome
        pass

    def __build_sequence(self, active_genes):
        pass

    def mutate(self):
        pass

    def get_active_genes(self):
        pass
