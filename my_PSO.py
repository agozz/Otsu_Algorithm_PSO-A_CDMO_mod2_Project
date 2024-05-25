import numpy as np
import random as rnd


class Particle():

    def __init__(self, space):
        self.position = 128 + rnd.randint(-32, 32)
        self.pbest_position = self.position
        self.pbest_value = space.fitness(self.position)
        self.velocity = 0


class Space():

    def __init__(self, image, n_particles):

        self.n_particles = n_particles
        self.image = image
        self.histogram = self.histogram()
        self.gbest_position = 128
        self.gbest_value = self.fitness(self.gbest_position)
        self.particles = [Particle(self) for _ in range(self.n_particles)]

    def histogram(self):

        histogram = np.zeros(256, dtype=np.int32)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                value = self.image[i, j]

                histogram[value] += 1

        return histogram

    def fitness(self, t):

        N1 = sum(self.histogram[0:t - 1]) + 0.1
        N2 = sum(self.histogram[t:255]) + 0.1

        N = self.image.shape[0] * self.image.shape[1]

        q1 = (N1 / N) + 0.1
        q2 = (1 - q1) + 0.1

        mu1 = 0
        mu2 = 0

        sig_quadro1 = 0
        sig_quadro2 = 0

        for i in range(0, t):
            mu1 += i * (self.histogram[i] / N1)

        for j in range(t, 256):
            mu2 += j * (self.histogram[j] / N2)

        for i in range(0, t):
            sig_quadro1 += ((i - mu1) ** 2) * ((self.histogram[i] / N) * (1 / q1))

        for j in range(t, 256):
            sig_quadro2 += ((j - mu2) ** 2) * ((self.histogram[j] / N) * (1 / q2))

        sig_quadroW = q1 * sig_quadro1 + q2 * sig_quadro2

        return sig_quadroW

    def set_gbest(self):

        for particle in self.particles:
            if self.gbest_value > particle.pbest_value:
                self.gbest_value = particle.pbest_value
                self.gbest_position = particle.position

    def search_PSO(self, w=0.6, c1=0.3, c2=0.9):

        for _ in range(5):

            self.set_gbest()

            for particle in self.particles:

                new_velocity = round(
                    w * particle.velocity + ((c1 * rnd.random()) * (particle.pbest_position - particle.position)) + \
                    ((c2 * rnd.random()) * (self.gbest_position - particle.position)))

                particle.position = particle.position + new_velocity
                particle.velocity = new_velocity

                fitness_candidate = self.fitness(particle.position)

                if particle.pbest_value > fitness_candidate:
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position = particle.position

        return self.gbest_position
