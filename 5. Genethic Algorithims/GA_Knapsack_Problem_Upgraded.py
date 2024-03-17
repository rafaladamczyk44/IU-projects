import random

# Set the global variables and parameters
K_CAP = 30  # max. knapsack capacity
POPULATION_SIZE = 30  # maximum size of population in a generation
MAX_GENERATION = 50  # number of maximum generations allowed
NUM_ITEMS = 15  # how many items do we consider in each run
CROSSOVER_RATE = 0.8  # crossover rate
MUTATION_RATE = 0.1  # initial mutation rate
MAX_STAGNATION = 10  # maximum generations without improvement to stop the algorithm
ITEMS = [(random.randint(0, 20), random.randint(0, 20)) for _ in range(NUM_ITEMS)]


def greedy_initialization(population_size, items, knapsack_capacity):
    """
    Initialize the population using a greedy heuristic approach.

    Args:
        population_size (int): The size of the population.
        items (list): The list of items and their values and weights.
        knapsack_capacity (int): The maximum capacity of the knapsack.

    Returns:
        list: A list containing the initial population.
    """
    population = []
    for _ in range(population_size):
        individual = [0] * len(items)
        capacity = knapsack_capacity
        for i in range(len(items)):
            if items[i][1] <= capacity:
                individual[i] = 1
                capacity -= items[i][1]
        population.append(individual)
    return population


def tournament_selection(population, tournament_size):
    """
    Perform tournament selection to choose parents for crossover.

    Args:
        population (list): The current population.
        tournament_size (int): The size of the tournament.

    Returns:
        list: A list containing the selected parents.
    """
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: population_fitness(ind, ITEMS, K_CAP))
        selected_parents.append(winner)
    return selected_parents


def two_point_crossover(parent1, parent2):
    """
    Perform two-point crossover to generate children.

    Args:
        parent1 (list): The first parent.
        parent2 (list): The second parent.

    Returns:
        tuple: A tuple containing the two children.
    """
    crossover_points = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:]
    child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[crossover_points[1]:]
    return child1, child2


def mutation(child):
    """
    Perform mutation on the child.

    Args:
        child (list): The child individual.

    Returns:
        list: The mutated child.
    """
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] = 1 if child[i] == 0 else 0
    return child


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
    for i in range(len(individual)):
        if individual[i] == 1:
            total_value += items_checked[i][0]
            total_weight += items_checked[i][1]
    if total_weight > knapsack_capacity:
        return 0
    else:
        return total_value


def evolution(population):
    """
    Evolve the population through selection, crossover, and mutation.

    Args:
        population (list): The current population to evolve.

    Returns:
        list: The evolved population.
    """
    selected_parents = tournament_selection(population, 3)
    new_population = []
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        if random.random() < CROSSOVER_RATE:
            child1, child2 = two_point_crossover(parent1, parent2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            new_population.extend([child1, child2])
        else:
            new_population.extend([parent1, parent2])
    return new_population


def is_stagnant(best_fitness_history):
    """
    Check if the algorithm is stagnant based on the improvement in best fitness.

    Args:
        best_fitness_history (list): History of the best fitness values.

    Returns:
        bool: True if the algorithm is stagnant, False otherwise.
    """
    if len(best_fitness_history) < MAX_STAGNATION:
        return False
    else:
        return len(set(best_fitness_history[-MAX_STAGNATION:])) == 1


# Initialize the population
population = greedy_initialization(POPULATION_SIZE, ITEMS, K_CAP)
best_fitness_history = []

# Main loop for the genetic algorithm
for gen in range(MAX_GENERATION):
    print("Generation:", gen)

    # Calculate fitness for each individual in the population
    fitness_values = [population_fitness(individual, ITEMS, K_CAP) for individual in population]

    # Track the best individual and its fitness
    best_individual = population[fitness_values.index(max(fitness_values))]
    best_fitness = max(fitness_values)
    best_fitness_history.append(best_fitness)

    print("Best Fitness:", best_fitness)

    # Check for stagnation
    if is_stagnant(best_fitness_history):
        print("Algorithm stagnated. Stopping early.")
        break

    # Evolve the population
    population = evolution(population)

    gen += 1

print("Best Solution:", best_individual)
