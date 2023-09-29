# Used to help abstract the genetic algorithm
import random
import uuid


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

        if len_per_seg == 1:
            sequence_segments = list(map(lambda x: {x[0]: x[1]}, self.sequence.items()))
            return sequence_segments

        i = 0
        n = 0
        tmp = dict()
        # for entry in self.sequence:
        #     if i < len_per_seg:
        #         tmp[entry] = self.sequence[entry]
        #         i += 1
        #     else:
        #         seg.append(tmp)
        #         tmp = dict()
        #         i = 0
        # if len(tmp) > 0:
        #     seg.append(tmp)

        for entry in self.sequence:
            if i > len_per_seg:
                sequence_segments.append(tmp)
                tmp = dict()
                i = 0

            tmp[entry] = self.sequence[entry]
            i += 1
        if len(tmp) > 0:
            sequence_segments.append(tmp)

        return sequence_segments

    def mutate_via_bitmask(self, mutation_rate=0.0, bit_mask=None):

        if bit_mask is None:
            bit_mask = []
            for i in range(0, len(Chromosome.genes)):
                if mutation_rate > random.random():
                    bit_mask.append(1)
                else:
                    bit_mask.append(0)

        i = 0
        for gene in self.sequence:
            self.sequence[gene] = bit_mask[i] ^ self.sequence[gene]
            i += 1

    def get_active_genes(self):
        return list(filter(lambda x: self.sequence[x] == 1, self.sequence))

    def __str__(self):
        return (f'\nChromosome ID: {self.genetic_id} \n'
                f'Fitness: {self.fitness} \n'
                f'----------------------------------- \n'
                f'{self.sequence}')


def crossover_chromosomes(a: Chromosome, b: Chromosome, bit_mask):
    a_segs = a.get_gene_segments()
    b_segs = b.get_gene_segments()
    child = None  # New Chromosome
    new_flag_list = []

    if len(a_segs) != len(b_segs):
        raise RuntimeError("How are they different lengths? This is not good.")

    for i in range(0, len(a_segs)):
        if bit_mask[i] == 0:
            new_flag_list += list(filter(lambda entry: a_segs[i][entry] == 1, a_segs[i]))
        else:
            new_flag_list += list(filter(lambda entry: b_segs[i][entry] == 1, b_segs[i]))

    child = Chromosome(new_flag_list, uuid.uuid4())

    return child
