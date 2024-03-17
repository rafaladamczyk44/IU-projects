import random

# Set the global variables

K_CAP = 30  # max. knapsack capacity
POPULATION_SIZE = 30  # maximum size of population in a generation
MAX_GENERATION = 50  # number of maximum generations allowed
NUM_ITEMS = 15  # how many items do we consider in each run
# Generate random items with random values and weights
ITEMS = [(random.randint(0, 20), random.randint(0, 20)) for x in range(0, NUM_ITEMS)]

generation = 1
print(ITEMS)

# Function for initializing the population
def initialization(population_size, number_of_items):
    """
    Initialize the population with random individuals.

    Args:
        population_size (int): The size of the population.
        number_of_items (int): The number of items each individual has.

    Returns:
        list: A list containing the initial population.
    """
    print("Initialization started")
    population = []
    for i in range(0, population_size):
        individual = []
        # Generate a random individual with 'number_of_items' genes
        for j in range(0, number_of_items):
            individual.append(random.randint(0, 1))
        population.append(individual)

    print("Initialization finished")
    print("Population Size: " + str(population_size))

    return population

# Function to calculate the fitness of an individual in the population
def population_fitness(individual, items_checked, knapsack_capacity):
    """
    Calculate the fitness of an individual based on its total value and weight.

    Args:
        individual (list): The individual for which to calculate the fitness.
        items_checked (list): The list of items and their values and weights.
        knapsack_capacity (int): The maximum capacity of the knapsack.

    Returns:
        int: The fitness value of the individual.
    """
    total_value = 0
    total_weight = 0
    index = 0
    for i in individual:
        if index >= len(items_checked):
            break

        if i == 1:
            total_value += items_checked[index][0]  # Add value if the item is chosen
            total_weight += items_checked[index][1]  # Add weight if the item is chosen
        index += 1
    # Check if the total weight exceeds the knapsack capacity
    if total_weight > knapsack_capacity:
        return 0
    else:
        return total_value

# Function for the evolution of the population
def population_evolution(population):
    """
    Evolve the population through selection, crossover, and mutation.

    Args:
        population (list): The current population to evolve.

    Returns:
        list: The evolved population.
    """
    parent_percent = 0.2
    mutation_chance = 0.08
    parent_lottery = 0.05

    # Selection: Select the best individuals to be parents
    parent_length = int(parent_percent * len(population))
    parent = population[:parent_length]

    # Crossover: Create children by combining genes of parents
    children = []
    desired_length = len(population) - len(parent)
    while len(children) < desired_length:
        male = population[random.randint(0, len(parent) - 1)]
        female = population[random.randint(0, len(parent) - 1)]
        half_of_parent = int(len(male) / 2)
        child = male[:half_of_parent] + female[half_of_parent:]
        children.append(child)  # Append the child to the children list

    # Mutation: Introduce random changes to some genes
    for child in children:
        if mutation_chance > random.random():
            r = random.randint(0, len(child) - 1)
            if child[r] == 1:
                child[r] = 0
            else:
                child[r] = 1

    # Combine parents and children to form the new population
    parent.extend(children)
    return parent

# Initialize the population
population = initialization(POPULATION_SIZE, NUM_ITEMS)

# Main loop for the genetic algorithm
for gen in range(0, MAX_GENERATION):
    print("Generation: ", gen)

    # Sort the population based on fitness
    population = sorted(population, key=lambda ind: population_fitness(ind, ITEMS, K_CAP), reverse=True)
    total_fitness = 0

    # Calculate total fitness
    for individual in population:
        total_fitness += population_fitness(individual, ITEMS, K_CAP)

    print("Total fitness: ", total_fitness)

    # Evolve the population
    population = population_evolution(population)
    gen += 1

    # Again sort the population based on fitness
    population = sorted(population, key=lambda ind: population_fitness(ind, ITEMS, K_CAP), reverse=True)
    print(population[0])
