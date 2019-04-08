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
num_generations = 20
gene_deviation = 16
location_mute = (-4, 4)  # define borders for location mutation
alpha = 0.5
src_filename = "image_test/mona.jpg"


class Point:

    def __init__(self, x=None, y=None, rand=False, from_point=None):
        if from_point is not None:
            x = int(from_point.x + random.randint(-gene_deviation, gene_deviation + 1))
            if x < 0:
                x = 0
            if x > IMAGE_SIZE[0]:
                x = IMAGE_SIZE[0]
            y = int(from_point.y + random.randint(-gene_deviation, gene_deviation + 1))
            if y < 0:
                y = 0
            if y > IMAGE_SIZE[1]:
                y = IMAGE_SIZE[1]
        elif rand:
            x = random.randint(0, IMAGE_SIZE[0]-1)
            y = random.randint(0, IMAGE_SIZE[0]-1)
        self.x = x
        self.y = y

    @staticmethod
    def get_center(*points, location=None):
        if location is not None:
            points = location.points
        x = sum(map(lambda point: point.x, points)) // len(points)
        y = sum(map(lambda point: point.y, points)) // len(points)
        if x > 255:
            x = 255
        if y > 255:
            y = 255
        return Point(x, y)


class Color:

    def __init__(self, r=None, g=None, b=None):
        if r is None:
            r, g, b = [random.randint(0, 255) for _ in range(3)]
        self.r = r
        self.g = g
        self.b = b

    def pure(self):
        return self.r, self.g, self.b


class Location:

    def __init__(self, points=None):
        if points is not None:
            self.points = points
        else:
            self.points = list()

    def __iadd__(self, point):
        assert type(point) == Point, 'Can not add non-point'
        self.points.append(point)
        return Location(points=self.points)

    def __getitem__(self, key):
        return self.points[key]

    def pure(self):
        return [(point.y, point.x) for point in self.points]


class Gene:

    def __init__(self):
        self.color = Color()
        self.location = Location()

        center_point = Point(rand=True)  # random point
        for _ in range(3):
            point = Point(from_point=center_point)
            self.location += point

    def mutate(self):

        center = Point.get_center(location=self.location)

        if random.random() < mutation_chance:
            # mutate color
            r, g, b = image[center.x, center.y]
            self.color = Color(int(r), int(g), int(b))

        if random.random() < mutation_chance:
            # mutate location
            location = Location()

            for i in range(3):
                # mutate one point
                pt = Point()
                pt.y = self.location[i].y + random.randint(*location_mute)
                if pt.y < 0:
                    pt.y = 0
                if pt.y > IMAGE_SIZE[0]:
                    pt.y = IMAGE_SIZE[0]
                elif pt.y - center.y > gene_deviation:
                    pt.y = center.y + gene_deviation
                elif pt.y - center.y < -gene_deviation:
                    pt.y = center.y - gene_deviation

                pt.x = self.location[i].x + random.randint(*location_mute)
                if pt.x < 0:
                    pt.x = 0
                if pt.x > IMAGE_SIZE[1]:
                    pt.x = IMAGE_SIZE[1]
                elif pt.x - center.x > gene_deviation:
                    pt.x = center.x + gene_deviation
                elif pt.x - center.x < -gene_deviation:
                    pt.x = center.x - gene_deviation

                location += pt
            self.location = location

    def draw(self, overlay):
        cv2.drawContours(overlay, [np.array(self.location.pure())], 0, self.color.pure(), -1)


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
        self.population = [Chromosome(), Chromosome()]

    def generate_population(self):
        for generation in tqdm(range(num_generations), unit='generation'):
            children = [self.population[0] + self.population[1] for _ in range(3)]
            self.population = sorted(children, key=lambda x: x.fitness)[:2]
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
    out_filename = src_filename.split('.')[0] + '_result.jpg'
    cv2.imwrite(out_filename, im)
    print(f'Done. Check result in {out_filename}')


if __name__ == '__main__':

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
