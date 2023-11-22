#!/bin/env python3

import pandas as pd
import matplotlib.pyplot as matplot

# a)
print("a)")
data = pd.read_csv("miasta.csv")
print(data)
print(data.values)

# b)
print("b)")
data.loc[len(data)] = [2010, 460, 555, 405]
print(data)

# c)
print("c)")
fig, axs = matplot.subplots()
axs.set_xlabel("Lata")
axs.set_ylabel("Liczba ludności (w tys.)")
axs.set_title("Ludność w miastach Polski")
axs.tick_params(axis='y', labelrotation=90)
axs.set_xticks(data.index)
axs.set_xticklabels(data["Rok"])
axs.set_yticks(range(150, 500, 50))
matplot.plot(data.index, data["Gdansk"], marker='o', color='r', markerfacecolor='none')
matplot.show()

# d)
print("d)")
fig, axs = matplot.subplots()
axs.set_xlabel("Lata")
axs.set_ylabel("Liczba ludności (w tys.)")
axs.set_title("Ludność w miastach Polski")
axs.tick_params(axis='y', labelrotation=90)
axs.set_xticks(data.index)
axs.set_xticklabels(data["Rok"])
axs.set_yticks(range(100, 650, 50))

cities = ["Gdansk", "Poznan", "Szczecin"]
colors = ["orange", "purple", "teal"]

for city, color in zip(cities, colors):
        matplot.plot(data.index, data[city], marker='o', color=color, markerfacecolor='none')

matplot.legend(cities, loc="upper left")
matplot.show()
