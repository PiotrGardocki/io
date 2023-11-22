import math

def activate_function(x):
    return 1 / (1 + math.pow(math.e, -x))

neural_network = (
    ((-0.46122, 0.97314, -0.39203, 0.80109),
     (0.78548, 2.10584, -0.57847, 0.43529)),
    ((-0.81546, 1.03775, -0.2368),)
)

funcs = (activate_function, lambda x: x)

def forwardPass(age, weight, height):
    numbers = [age, weight, height]

    for i, neurons in enumerate(neural_network):
        new_numbers = []

        for weigths in neurons:
            sum = 0
            for n1, n2 in zip(numbers, weigths):
                sum += n1 * n2
            sum += weigths[-1]
            new_numbers.append(funcs[i](sum))

        numbers = new_numbers

    return numbers[0]

print("Zadanie 1")
print(forwardPass(23, 75, 176))
