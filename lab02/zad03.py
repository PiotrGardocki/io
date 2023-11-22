#!/bin/env python3

import pygad
import math

def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + \
           math.sin(z * u) + math.cos(v * w)

def fitness_func(ga_instance, solution, solution_idx):
    return endurance(*solution)

gene_space = {'low': 0, 'high': 1}
fitness_function = fitness_func

sol_per_pop = 10
num_genes = 6

num_parents_mating = 5
num_generations = 100
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = round(100 / num_genes)

ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution: {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = endurance(*solution)
print("Predicted output based on the best solution: {prediction}".format(prediction=prediction))

ga_instance.plot_fitness()
