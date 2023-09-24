# Used to help abstract the genetic algorithm

class Chromosome:
    genes = []

    def __init__(self, active_genes, id):
        self.sequence = dict()  # 00100100
        self.genetic_id = id
        # 1 - flag is enabled in chromosome
        # 0 - flag is disabled in chromosome
        self.__build_sequence(active_genes)
        self.fitness = -1

    def __build_sequence(self, active_genes):

        self.sequence["-O0"] = 1  # Always have -O0 enabled at the start.

        for g in Chromosome.genes:
            if g in active_genes:
                self.sequence[g] = 1
            else:
                self.sequence[g] = 0

    def mutate(self):
        pass

    def get_active_genes(self):
        return list(filter(lambda x: self.sequence[x] == 1, self.sequence))

    def __str__(self):
        return (f'\nChromosome ID: {self.genetic_id} \n'
                f'Fitness: {self.fitness} \n'
                f'----------------------------------- \n'
                f'{self.sequence}')
