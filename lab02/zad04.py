#!/bin/env python3

import pygad
import time

maze = [
    '111111111111',
    '100010001001',
    '111000101101',
    '100010100001',
    '101011001101',
    '100110001001',
    '100000100011',
    '101001101001',
    '101110001101',
    '101011010101',
    '101000000001',
    '111111111111',
]

maze = [
    [bool(int(c)) for c in row] for row in maze
]

start_pos = (1, 1)
end_pos = (10, 10)

directions = [
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0)
]

def get_end_pos(solution):
    pos = start_pos

    for step in solution:
        d = directions[int(step)]
        new_pos = (pos[0] + d[0], pos[1] + d[1])
        if not maze[new_pos[1]][new_pos[0]]:
            pos = new_pos

    return pos

def fitness_func(ga_instance, solution, solution_idx):
    pos = get_end_pos(solution)
    length = abs(pos[0] - end_pos[0]) + abs(pos[1] - end_pos[1])
    return -length

gene_space = list(range(len(directions)))
fitness_function = fitness_func

sol_per_pop = 100
num_genes = 30

num_parents_mating = 15
num_generations = 500
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = round(100 / num_genes) + 1

times = []

for _ in range(10):
    start = time.time()

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
                           mutation_percent_genes=mutation_percent_genes,
                           stop_criteria='reach_0')

    ga_instance.run()

    end = time.time()
    times.append(end - start)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution: {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = get_end_pos(solution)
print("Predicted output based on the best solution: {prediction}".format(prediction=prediction))

#ga_instance.plot_fitness()

print("Czasy wykonania:")
for t in times:
    print(f"{t * 1000}ms")
print(f"Średni czas: {sum(times) * 1000 / len(times)}ms")
