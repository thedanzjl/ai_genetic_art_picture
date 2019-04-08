import cv2
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import argparse


#  Definitions
IMAGE_SIZE = (256, 256)
mutation_chance = 0.2
num_genes = 2500
num_generations = 50
gene_deviation = 16
location_mute = (-4, 4)  # define borders for location mutation
alpha = 0.5


class Gene:

    def __init__(self):
        self.color = [random.randint(0, 255) for _ in range(3)]
        x, y = (random.randint(0, IMAGE_SIZE[0]), random.randint(0, IMAGE_SIZE[1]))
        self.location = list()
        for _ in range(3):
            ptx = int(x + random.randint(-gene_deviation, gene_deviation+1))
            if ptx < 0:
                ptx = 0
            if ptx > IMAGE_SIZE[0]:
                ptx = IMAGE_SIZE[0]
            pty = int(y + random.randint(-gene_deviation, gene_deviation+1))
            if pty < 0:
                pty = 0
            if pty > IMAGE_SIZE[1]:
                pty = IMAGE_SIZE[1]
            self.location.append((ptx, pty))

    def mutate(self):

        y = (self.location[0][0] + self.location[1][0] + self.location[2][0]) // 3
        x = (self.location[0][1] + self.location[1][1] + self.location[2][1]) // 3

        if random.random() < mutation_chance:
            # mutate color
            if x > 255:
                x = 255
            if y > 255:
                y = 255
            r, g, b = image[x, y]
            self.color = (int(r), int(g), int(b))

        if random.random() < mutation_chance:
            # mutate location
            pt1 = [0, 0]
            pt1[0] = self.location[0][0] + random.randint(*location_mute)
            if pt1[0] < 0:
                pt1[0] = 0
            if pt1[0] > IMAGE_SIZE[0]:
                pt1[0] = IMAGE_SIZE[0]
            elif pt1[0] - y > gene_deviation:
                pt1[0] = y + gene_deviation
            elif pt1[0] - y < -gene_deviation:
                pt1[0] = y - gene_deviation

            pt1[1] = self.location[0][1] + random.randint(*location_mute)
            if pt1[1] < 0:
                pt1[1] = 0
            if pt1[1] > IMAGE_SIZE[1]:
                pt1[1] = IMAGE_SIZE[1]
            elif pt1[1] - x > gene_deviation:
                pt1[1] = x + gene_deviation
            elif pt1[1] - x < -gene_deviation:
                pt1[1] = x - gene_deviation

            pt2 = [0, 0]
            pt2[0] = self.location[1][0] + random.randint(*location_mute)
            if pt2[0] < 0:
                pt2[0] = 0
            if pt2[0] > IMAGE_SIZE[0]:
                pt2[0] = IMAGE_SIZE[0]
            elif pt2[0] - y > gene_deviation:
                pt2[0] = y + gene_deviation
            elif pt2[0] - y < -gene_deviation:
                pt2[0] = y - gene_deviation

            pt2[1] = self.location[1][1] + random.randint(*location_mute)
            if pt2[1] < 0:
                pt2[1] = 0
            if pt2[1] > IMAGE_SIZE[1]:
                pt2[1] = IMAGE_SIZE[1]
            elif pt2[1] - x > gene_deviation:
                pt2[1] = x + gene_deviation
            elif pt2[1] - x < -gene_deviation:
                pt2[1] = x - gene_deviation

            pt3 = [0, 0]
            pt3[0] = self.location[2][0] + random.randint(*location_mute)
            if pt3[0] < 0:
                pt3[0] = 0
            if pt3[0] > IMAGE_SIZE[0]:
                pt3[0] = IMAGE_SIZE[0]
            elif pt3[0] - y > gene_deviation:
                pt3[0] = y + gene_deviation
            elif pt3[0] - y < -gene_deviation:
                pt3[0] = y - gene_deviation

            pt3[1] = self.location[2][1] + random.randint(*location_mute)
            if pt3[1] < 0:
                pt3[1] = 0
            if pt3[1] > IMAGE_SIZE[1]:
                pt3[1] = IMAGE_SIZE[1]
            elif pt3[1] - x > gene_deviation:
                pt3[1] = x + gene_deviation
            elif pt3[1] - x < -gene_deviation:
                pt3[1] = x - gene_deviation
        else:
            pt1, pt2, pt3 = self.location

        self.location = [tuple(pt1), tuple(pt2), tuple(pt3)]

    def draw(self, overlay):
        cv2.drawContours(overlay, [np.array(self.location)], 0, self.color, -1)


class Chromosome:

    def __init__(self, genes=None):
        if genes is None:
            self.genes = [Gene() for _ in range(num_genes)]
        else:
            self.genes = genes
        self.fitness = self.get_fitness()

    def get_fitness(self):  # less fitness -> better
        diff = np.sum((image.astype("float") - self.get_image().astype("float")) ** 2)
        diff /= float(IMAGE_SIZE[0] * IMAGE_SIZE[1])
        return diff

    def __add__(self, other):  # crossover
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
        self.population = [Chromosome() for _ in range(2)]

    def generate_population(self):
        for generation in tqdm(range(num_generations), unit='generation'):
            child1 = self.population[0] + self.population[1]
            child2 = self.population[0] + self.population[1]
            child3 = self.population[0] + self.population[1]
            if child3.fitness < child2.fitness and child3.fitness < child1.fitness:
                self.population = [child1, child2]
            elif child1.fitness < child2.fitness and child1.fitness < child3.fitness:
                self.population = [child2, child3]
            elif child2.fitness < child1.fitness and child2.fitness < child3.fitness:
                self.population = [child1, child3]
            yield self.population

    @staticmethod
    def best_chromosome(population):
        result = min(population, key=lambda x: x.fitness)
        return result


def main():
    print(f'working on {src_filename}...')
    ai = GeneticImage()
    for population in ai.generate_population():
        best_art = GeneticImage.best_chromosome(population)
        im = best_art.get_image()
        cv2.imshow(" ", im)
        cv2.waitKey(1)

    im = cv2.resize(im, (512, 512))
    cv2.imwrite('result', im)
    cv2.waitKey()


if __name__ == '__main__':

    src_filename = "image_test/loli.jpg"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--generations")
    args = parser.parse_args()
    if args.path is not None:
        src_filename = args.path
    if args.generations is not None:
        num_generations = args.generations

    image = cv2.imread(src_filename)
    image = cv2.resize(image, IMAGE_SIZE)
    main()
