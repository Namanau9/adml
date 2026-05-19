import numpy as np
import random
import matplotlib.pyplot as plt


def fitness(x):
    return x**2


def initialize_population(pop_size, lower, upper):
    return np.random.uniform(lower, upper, pop_size)


def select_parents(population, fitness_values):

    parents = []

    for _ in range(len(population) // 2):

        tournament = random.sample(list(zip(population, fitness_values)), 3)

        parents.append(max(tournament, key=lambda x: x[1])[0])

    return np.array(parents)


def crossover(parents):

    offspring = []

    for i in range(0, len(parents), 2):

        if i + 1 < len(parents):

            child = (parents[i] + parents[i + 1]) / 2

            offspring.append(child)

    return np.array(offspring)


def mutate(population, mutation_rate, lower, upper):

    for i in range(len(population)):

        if random.random() < mutation_rate:

            population[i] += np.random.uniform(-0.5, 0.5)

            population[i] = np.clip(population[i], lower, upper)

    return population


def genetic_algorithm(pop_size, generations, lower, upper, mutation_rate):

    population = initialize_population(pop_size, lower, upper)

    fitness_history = []

    for generation in range(generations):

        fitness_values = np.array([fitness(x) for x in population])

        if len(fitness_values) == 0:

            print(f"Error: Population has size zero at generation {generation}")

            break

        best_fitness = np.max(fitness_values)

        fitness_history.append(best_fitness)

        if generation < 3 or generation == 99:

            print(f"Generation {generation + 1} - Best Fitness: {best_fitness}")

            print(f"Generation {generation + 1} - Population Size: {len(population)}")

        parents = select_parents(population, fitness_values)

        offspring = crossover(parents)

        while len(offspring) < len(population):

            offspring = np.append(
                offspring,
                parents[random.randint(0, len(parents) - 1)]
            )

        population = mutate(offspring, mutation_rate, lower, upper)

        if len(population) == 0:

            print("Warning: Population became empty. Reinitializing...")

            population = initialize_population(pop_size, lower, upper)

    fitness_values = np.array([fitness(x) for x in population])

    best_solution = population[np.argmax(fitness_values)]

    return best_solution, fitness_history


best_solution, fitness_history = genetic_algorithm(
    pop_size=20,
    generations=100,
    lower=-5,
    upper=5,
    mutation_rate=0.1
)

print(f"Best solution: {best_solution}")

print(f"Best fitness (f(x) = x^2): {fitness(best_solution)}")


plt.plot(fitness_history)

plt.xlabel('Generation')

plt.ylabel('Best Fitness Value')

plt.title('Fitness Progression Over Generations')

plt.show()
