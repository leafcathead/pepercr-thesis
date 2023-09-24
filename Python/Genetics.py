# Used to help abstract the genetic algorithm

class Chromosome:
    genes = []
    num_of_segments = 0

    def __init__(self, active_genes, g_id):
        self.sequence = dict()  # 00100100
        self.genetic_id = g_id

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

    def get_gene_segments(self):
        if Chromosome.num_of_segments == 0:
            raise ZeroDivisionError("Did you configure the number of segments correctly in the configuration file?")

        len_per_seg = round(len(Chromosome.genes) / Chromosome.num_of_segments)
        sequence_segments = []

        i = 0
        n = 0
        seg = []
        tmp = dict()
        for entry in self.sequence:
            if i < len_per_seg:
                tmp[entry] = self.sequence[entry]
                i += 1
            else:
                seg.append(tmp)
                tmp = dict()
                i = 0

        print(seg)

        return seg


    def mutate(self):
        pass

    def get_active_genes(self):
        return list(filter(lambda x: self.sequence[x] == 1, self.sequence))

    @staticmethod
    def crossover_chromosomes(a, b):
        pass

    def __str__(self):
        return (f'\nChromosome ID: {self.genetic_id} \n'
                f'Fitness: {self.fitness} \n'
                f'----------------------------------- \n'
                f'{self.sequence}')
