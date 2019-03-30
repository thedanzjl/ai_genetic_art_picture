import cv2
import random
import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm


#  Definitions
IMAGE_SIZE = (256, 256)
mutation_chance = 0.1
num_population = 50
num_genes = 70
num_generations = 10000
k_best = 0.15
mute = 0.1
alpha = 0.5
src_filename = "mona.jpg"


image = cv2.imread(src_filename)
image = cv2.resize(image, IMAGE_SIZE)


class EllipseGene:

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
        if random.random() < mute:
            self.color = (self.color[0] + random.randint(*color_mute),
                          self.color[1] + random.randint(*color_mute), self.color[2] + random.randint(*color_mute))

        if random.random() < mute:
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
        else:
            center = self.location[0]

        if random.random() < mute:
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
        else:
            axis = self.location[1]

        if random.random() < mute:
            angel = self.location[2] + random.randint(*angel_mute)
        else:
            angel = self.location[2]

        self.location = [tuple(center), tuple(axis), angel, self.location[3], self.location[4]]

    def draw(self, overlay):
        cv2.ellipse(overlay, *self.location, self.color, -1)


class CircleGene:

    def __init__(self):
        center = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        radius = random.randint(int(IMAGE_SIZE[0]/8), int(IMAGE_SIZE[0]/4))
        self.color = [random.randint(0, 255) for _ in range(3)]
        self.location = [
            center, radius
        ]

    def mutate(self):
        color_mute = (int(-256 * mute), int(256 * mute))
        radius_mute = (int(-512 * mute), int(512 * mute))
        location_mute = (int(-IMAGE_SIZE[0] * mute), int(IMAGE_SIZE[1] * mute))

        if random.random() < mute:
            self.color = (self.color[0] + random.randint(*color_mute),
                          self.color[1] + random.randint(*color_mute), self.color[2] + random.randint(*color_mute))

        if random.random() < mute:
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
        else:
            center = self.location[0]

        if random.random() < mute:
            radius = self.location[1] + random.randint(*radius_mute)
            if radius < 0:
                radius = 0
            if radius > IMAGE_SIZE[0]:
                radius = IMAGE_SIZE[0]
        else:
            radius = self.location[1]

        self.location = [tuple(center), radius]

    def draw(self, overlay):
        cv2.circle(overlay, *self.location, self.color, -1)


class TriangleGene:

    def __init__(self):
        self.color = [random.randint(0, 255) for _ in range(3)]
        x, y = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        self.location = list()
        for _ in range(3):
            ptx = int(x + random.randint(0, IMAGE_SIZE[0]) - IMAGE_SIZE[0]/2)
            if ptx < 0:
                ptx = 0
            if ptx > IMAGE_SIZE[0]:
                ptx = IMAGE_SIZE[0]
            pty = int(y + random.randint(0, IMAGE_SIZE[1]) - IMAGE_SIZE[1]/2)
            if pty < 0:
                pty = 0
            if pty > IMAGE_SIZE[1]:
                pty = IMAGE_SIZE[1]
            self.location.append((ptx, pty))

    def mutate(self):
        color_mute = (int(-256 * mute), int(256 * mute))
        location_mute = (int(-IMAGE_SIZE[0] * mute), int(IMAGE_SIZE[1] * mute))

        if random.random() < mute:
            self.color = (self.color[0] + random.randint(*color_mute),
                          self.color[1] + random.randint(*color_mute),
                          self.color[2] + random.randint(*color_mute))

        if random.random() < mute:
            pt1 = [0, 0]
            pt1[0] = self.location[0][0] + random.randint(*location_mute)
            if pt1[0] < 0:
                pt1[0] = 0
            if pt1[0] > IMAGE_SIZE[0]:
                pt1[0] = IMAGE_SIZE[0]

            pt1[1] = self.location[0][1] + random.randint(*location_mute)
            if pt1[1] < 0:
                pt1[1] = 0
            if pt1[1] > IMAGE_SIZE[1]:
                pt1[1] = IMAGE_SIZE[1]

            pt2 = [0, 0]
            pt2[0] = self.location[1][0] + random.randint(*location_mute)
            if pt2[0] < 0:
                pt2[0] = 0
            if pt2[0] > IMAGE_SIZE[0]:
                pt2[0] = IMAGE_SIZE[0]

            pt2[1] = self.location[1][1] + random.randint(*location_mute)
            if pt2[1] < 0:
                pt2[1] = 0
            if pt2[1] > IMAGE_SIZE[1]:
                pt2[1] = IMAGE_SIZE[1]

            pt3 = [0, 0]
            pt3[0] = self.location[2][0] + random.randint(*location_mute)
            if pt3[0] < 0:
                pt3[0] = 0
            if pt3[0] > IMAGE_SIZE[0]:
                pt3[0] = IMAGE_SIZE[0]

            pt3[1] = self.location[2][1] + random.randint(*location_mute)
            if pt3[1] < 0:
                pt3[1] = 0
            if pt3[1] > IMAGE_SIZE[1]:
                pt3[1] = IMAGE_SIZE[1]

        else:
            pt1, pt2, pt3 = self.location

        self.location = [tuple(pt1), tuple(pt2), tuple(pt3)]

    def draw(self, overlay):
        cv2.drawContours(overlay, [np.array(self.location)], 0, self.color, -1)


class Chromosome:

    def __init__(self, genes=None):
        if genes is None:
            self.genes = [TriangleGene() for _ in range(num_genes)]
        else:
            self.genes = genes
        self.fitness = self.get_fitness()

    def get_fitness(self):  # less fitness -> better
        diff = np.sum( (image.astype("float") - self.get_image().astype("float")) ** 2 )
        diff /= float(IMAGE_SIZE[0] * IMAGE_SIZE[1])
        return diff

    def __add__(self, other):  # mating
        genes = deepcopy(self.genes) \
            if random.choice([0, 1]) else deepcopy(other.genes)
        for gene in genes:
            gene.mutate()
        return Chromosome(genes=genes)

    def get_image(self):
        origin = np.full((*IMAGE_SIZE, 3), 255, np.uint8)
        for gene in self.genes:
            overlay = origin.copy()
            gene.draw(overlay)
            cv2.addWeighted(overlay, alpha, origin, 1-alpha, 0, origin)
        return origin


    def __lt__(self, other):
        return self.fitness < other.fitness


class GeneticImage:

    def __init__(self):
        self.population = [Chromosome() for _ in range(num_population)]

    def generate_population(self):
        for generation in tqdm(range(num_generations), unit='generation'):
            old_population = self.population
            old_population.sort(reverse=False)
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
                    result_population.append(child)

            self.population = result_population
            yield result_population

    @staticmethod
    def best_chromosome(population):
        result = min(population, key=lambda x: x.fitness)
        # print(result.fitness)
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
