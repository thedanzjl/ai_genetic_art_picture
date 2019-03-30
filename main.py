import cv2
import random
import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm


#  Definitions
IMAGE_SIZE = (512, 512)
mutation_chance = 0.02
num_population = 35
num_genes = 60
num_generations = 10000
k_best = 0.15
mute = 0.15
src_filename = "mona.jpg"

image = cv2.imread(src_filename)
image = cv2.resize(image, IMAGE_SIZE)


class Gene:

    def __init__(self):
        # [ center, axis, angel, startAngel, endAngel ]
        center = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        angel = random.randint(0, 180)
        axis = [random.randint(IMAGE_SIZE[0]/8, IMAGE_SIZE[0]/4)]
        axis.append(axis[0] + random.randint(-10, 10))
        self.location = [
            center,
            tuple(axis), angel, 0, 360
        ]
        self.color = [random.randint(0, 255) for _ in range(3)]

    def mutate(self):
        color_mute = (int(-256 * mute), int(256 * mute))
        location_mute = (int(-IMAGE_SIZE[0] * mute), int(IMAGE_SIZE[1] * mute))
        angel_mute = (int(-180 * mute), int(180 * mute))
        # for gene in self.genes:
        self.color = (self.color[0] + random.randint(*color_mute),
                      self.color[1] + random.randint(*color_mute), self.color[2] + random.randint(*color_mute))
        # [ (0, 1), (2, 3), 4, 5, 6 ]

        center = [0, 0]
        center[0] = self.location[0][0] + random.randint(*location_mute)
        if center[0] < 0:
            center[0] = 0
        if center[0] > IMAGE_SIZE[0]:
            center[0] = IMAGE_SIZE[0]

        center[1] = self.location[0][1] + random.randint(*location_mute)
        if center[1] < 0:
            center[1] = 0
        if center[1] > IMAGE_SIZE[1]:
            center[1] = IMAGE_SIZE[1]

        axis = [0, 0]
        axis[0] = self.location[1][0] + random.randint(*location_mute)
        if axis[0] < 0:
            axis[0] = 0
        if axis[0] > IMAGE_SIZE[0]:
            axis[0] = IMAGE_SIZE[0]

        axis[1] = self.location[1][1] + random.randint(*location_mute)
        if axis[1] < 0:
            axis[1] = 0
        if axis[1] > IMAGE_SIZE[1]:
            axis[1] = IMAGE_SIZE[1]

        angel = self.location[2] + random.randint(*angel_mute)

        self.location = [tuple(center), tuple(axis), angel, self.location[3], self.location[4]]


class Chromosome:

    def __init__(self, genes=None):
        if genes is None:
            self.genes = [Gene() for _ in range(num_genes)]
        else:
            self.genes = genes
        self.fitness = self.get_fitness()

    def get_fitness(self):  # more fitness -> better
        diff = abs(np.sum(abs(image - self.get_image())))
        diff = 1 - diff / (IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3 * 256)
        return diff

    def __add__(self, other):  # mating
        genes = deepcopy(self.genes) \
            if random.choice([0, 1]) else deepcopy(other.genes)
        for gene in genes:
            if random.random() <= mutation_chance:
                gene.mutate()
        return Chromosome(genes=genes)

    def get_image(self):
        origin = np.full((*IMAGE_SIZE, 3), 0, np.uint8)
        alpha = 0.2
        for gene in self.genes:
            output = origin.copy()
            overlay = origin.copy()
            cv2.ellipse(overlay, *gene.location, gene.color, -1)
            cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
            origin = output
        return origin

    def __lt__(self, other):
        return self.fitness < other.fitness


class GeneticImage:

    def __init__(self):
        self.population = [Chromosome() for _ in range(num_population)]

    def generate_population(self):
        for generation in tqdm(range(num_generations)):
            old_population = self.population
            old_population.sort(reverse=True)
            num_best = math.floor(num_population * k_best)
            num_rand = math.ceil(1. / k_best)
            result_population = list()
            # noinspection PyTypeChecker
            for i in range(num_best):
                # noinspection PyTypeChecker
                for j in range(num_rand):
                    parent1 = old_population[i]
                    random_i = int(random.random() * num_best)
                    if random_i == i:  # parents should not be the same
                        random_i += 1
                    parent2 = old_population[random_i]
                    child = parent1 + parent2   # crossover
                    result_population.append(parent1)
                    result_population.append(child)

            self.population = result_population
            yield result_population

    @staticmethod
    def best_chromosome(population):
        result = max(population, key=lambda x: x.fitness)
        print(result.fitness)
        return result


def main():
    ai = GeneticImage()
    for population in ai.generate_population():
        best_art = GeneticImage.best_chromosome(population)
        im = best_art.get_image()
        cv2.imshow(" ", im)
        cv2.waitKey(1)

    cv2.waitKey()


if __name__ == '__main__':
    main()
