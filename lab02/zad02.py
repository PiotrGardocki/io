#!/bin/env python3

import pygad
import numpy
import time

prices = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
weigths = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
names = ['zegar', 'obraz-pejzaż', 'obraz-portret', 'radio',
         'laptop', 'lampka nocna', 'srebrne sztućce',
         'porcelana', 'figura z brązu', 'skórzana torebka',
         'odkurzacz']
max_weight = 25

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    sum_weigth = numpy.sum(solution * weigths)
    if sum_weigth > max_weight:
        return 0

    sum_price = numpy.sum(solution * prices)
    return sum_price

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(prices)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 100
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = round(100 / num_genes) + 1

times = []

for _ in range(10):
    start = time.time()

    #inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
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
                           stop_criteria=["reach_1600"])

    #uruchomienie algorytmu
    ga_instance.run()

    end = time.time()
    times.append(end - start)

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution: {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(prices * solution)
print("Predicted output based on the best solution: {prediction}".format(prediction=prediction))

print([names[idx] for idx, s in enumerate([bool(s) for s in solution]) if s])

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
#ga_instance.plot_fitness()

print("Czasy wykonania:")
for t in times:
    print(f"{t * 1000}ms")
print(f"Średni czas: {sum(times) * 1000 / len(times)}ms")
