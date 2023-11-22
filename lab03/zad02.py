import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")

num_coords = 20

#COORDS = tuple((random.randint(0,100), random.randint(0,100)) for _ in range(num_coords))
COORDS = ((39, 47), (71, 54), (90, 25), (29, 9), (21, 23), (40, 62), (17, 5), (100, 84), (3, 86), (88, 93), (71, 93), (96, 27), (87, 82), (34, 63), (55, 9), (35, 75), (34, 77), (70, 80), (75, 73), (8, 59))

def random_coord():
    r = random.randint(0, len(COORDS))
    return r

def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))

plot_nodes()

colony = AntColony(COORDS, ant_count=100, alpha=0.5, beta=1.2, 
                   pheromone_evaporation_rate=0.7, pheromone_constant=1000.0,
                   iterations=100)

# Ilość mrówek i iteracji najbardziej wpływa na długość trwania algorytmu
# Większe alpha szybciej daje lepsze wyniki
# Przy wysokim alpha beta ma niewielki wpływ na wyniki
# Małe alpha i beta mają spory rozrzut wyników przy małej ilości iteracji
# Zmiana pheromone_evaporation_rate na 0.7 znacząco poprawiła wyniki, a zmiana na 0.8 mocno pogorszyła.

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()