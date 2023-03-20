from functools import lru_cache
import random
import numpy as np

# Seems the start_pot is not defined, so I defined one
start_pot_coord = np.zeros(2)


def generate_cvrp_instance(num_customers, max_demand, max_capacity):
    customers = np.random.randint(1, max_demand, size=num_customers)
    coordinates = np.random.randint(0, 100, size=(num_customers, 2))
    return customers, coordinates, max_capacity


def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# TODO: implement the fitness evaluation function fitness()
def fitness(solution, customers, coordinates, max_capacity):
    '''
    @describe: calculate the fitness (total distance) of the solution

    @param: solution: a list of nodes, e.g., [3, 2, 4, 1]
    @param: customers: demand of each customer, e.g., [9, 7, 5, 7]
    @param: coordinates: coordinates of each customer, e.g., [array([85, 82]), array([80, 51]), array([95, 24]), array([49, 70]), array([ 5, 87])]
    @param: max_capacity: the capacity of each vehicle, e.g., 50

    @return: total distance of the solution
    '''
    tot_dist = 0
    cur_cap = max_capacity
    # Fill a guard element, which is the start pot
    guard_solution = [0] + solution
    guard_customers = np.insert(customers, 0, -1)
    guard_coordinates = np.insert(coordinates, 0, start_pot_coord, axis=0)
    for i, node in enumerate(guard_solution):
        if i == 0:
            continue

        if max_capacity < guard_customers[i]:
            raise ValueError("Current maximum capacity is not enough.")

        prev_pot = guard_solution[i - 1]
        if cur_cap - guard_customers[i] < 0:
            tot_dist += euclidean_distance(
                guard_coordinates[prev_pot], guard_coordinates[0])
            cur_cap = max_capacity
            prev_pot = 0
        tot_dist += euclidean_distance(
            guard_coordinates[prev_pot], guard_coordinates[node])
        cur_cap -= guard_customers[i]
    return tot_dist + euclidean_distance(guard_coordinates[solution[-1]], guard_coordinates[0])


def generate_initial_population(num_individuals, num_customers):
    population = []
    for _ in range(num_individuals):
        # individual = list(range(1, num_customers))
        # The original code above is not correct,
        # since it only generate individuals with dimension num_customers - 1
        individual = list(range(1, num_customers+1))
        random.shuffle(individual)
        population.append(individual)
    return population


# TODO: implement your mutation operator
def mutate(individual):
    '''
    @describe: the mutation operator
    @param: individual: a list of nodes, e.g., [3, 2, 4, 1]
    @return: individual after mutation
    '''
    pt1, pt2 = np.random.choice(len(individual), 2, replace=False)
    if pt1 > pt2:
        pt1, pt2 = pt2, pt1
    return individual[:pt1] + individual[pt1:pt2][::-1] + individual[pt2:]


# TODO: implement your crossover operator
def crossover(parent1, parent2):
    '''
    @describe: the crossover operator
    @param parent1: a list of nodes, e.g., [3, 2, 4, 1]
    @param parent2: a list of nodes, e.g., [4, 3, 2, 1]
    @return child1, child2: two offspring obtained by crossover
    '''
    assert len(parent1) == len(parent2)
    # Edge map
    edge_map = {i: set() for i in parent1}
    dim = len(parent1)
    for i in range(dim):
        edge_map[parent1[i]].add(parent1[(i-1) % dim])
        edge_map[parent1[i]].add(parent1[(i+1) % dim])
        edge_map[parent2[i]].add(parent2[(i-1) % dim])
        edge_map[parent2[i]].add(parent2[(i+1) % dim])
    # Generate children
    children = [[], []]
    for i in range(2):
        last = np.random.choice(list(edge_map.keys()), 1)[0]
        children[i].append(last)
        while len(children[i]) < dim:
            neighbors = edge_map[last]
            candidates = [neighbor for neighbor in neighbors if neighbor not in children[i]]
            # If all neighbors are already selected, select a new node to restart
            if candidates is None or len(candidates) == 0:
                candidates = [node for node in edge_map.keys() if node not in children[i]]
            min_num_neighbors = min(len(edge_map[k]) for k in candidates)
            candidates = [neighbor for neighbor in candidates if len(edge_map[neighbor]) == min_num_neighbors]
            last = np.random.choice(candidates, 1)[0]
            children[i].append(last)
    return children[0], children[1]


# TODO: implement your tournament selection method
def tournament_selection(population, customers, coordinates, max_capacity, alpha=10):
    '''
    @describe: the tournament selection method, select k individuals from polulation, the best one is selected as one parent
    @param: Determined by yourself
    @return parent
    '''
    ratio = alpha / 100.0
    race_size = int(ratio * len(population))
    win_cnt = [0] * len(population)
    for i, ind in enumerate(population):
        del population[i]
        chosen_pop = np.random.choice(
            len(population), size=race_size, replace=False)
        chosen_pop = np.asarray(population)[chosen_pop]
        for j, rival in enumerate(chosen_pop):
            fit_i, fit_j = fitness(ind, customers, coordinates, max_capacity), fitness(
                rival, customers, coordinates, max_capacity)
            win_prob = fit_j / (fit_i + fit_j)
            if np.random.random() <= win_prob:
                win_cnt[i] += 1
        population.insert(i, ind)
    top_winner = population[np.argmax(win_cnt)]
    return top_winner


def evolutionary_algorithm(num_generations, num_individuals, mutation_rate, num_customers, max_demand, max_capacity):
    customers, coordinates, capacity = generate_cvrp_instance(
        num_customers, max_demand, max_capacity)
    population = generate_initial_population(num_individuals, num_customers)
    best_solution = min(population, key=lambda x: fitness(
        x, customers, coordinates, capacity))
    best_fitness = fitness(best_solution, customers, coordinates, capacity)
    print("Best solution of current generation:", best_solution)
    print("Best fitness of current generation:", best_fitness)

    for generation in range(num_generations):
        print("Processing generation {}...".format(generation))
        new_population = []
        for _ in range(num_individuals // 2):
            # Selection
            parent1 = tournament_selection(
                population, customers, coordinates, max_capacity)
            parent2 = tournament_selection(
                population, customers, coordinates, max_capacity)
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            # Mutation
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population
        # Find the best solution in the current population
        best_solution = min(population, key=lambda x: fitness(
            x, customers, coordinates, capacity))
        best_fitness = fitness(best_solution, customers, coordinates, capacity)
        print("Best solution of current generation:", best_solution)
        print("Best fitness of current generation:", best_fitness)

    # Find the best solution in the final population
    best_solution = min(population, key=lambda x: fitness(
        x, customers, coordinates, capacity))
    best_fitness = fitness(best_solution, customers, coordinates, capacity)
    return best_solution, best_fitness


if __name__ == "__main__":
    num_generations = 100
    num_individuals = 50
    mutation_rate = 0.2
    num_customers = 20
    max_demand = 10
    max_capacity = 50

    best_solution, best_fitness = evolutionary_algorithm(
        num_generations, num_individuals, mutation_rate, num_customers, max_demand, max_capacity)

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
