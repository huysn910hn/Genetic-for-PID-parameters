import numpy as np
import control
import random
import matplotlib.pyplot as plt
from transfer_function import *

GENE_LENGTH = 8
GENERATIONS = 50
populationSize = 20
Pc = 0.8
Pm = 0.01
Kp_range = [0, 50]
Kd_range = [0, 10]
Ki_range = [0, (12+2*max(Kd_range))*(20.02+2*max(Kp_range))/2]


def encode_pid(Kp, Ki, Kd, Kp_range, Ki_range, Kd_range, bits=8):
    min_Kp, max_Kp = Kp_range
    min_Ki, max_Ki = Ki_range
    min_Kd, max_Kd = Kd_range
    Kp_bin = format(int((Kp - min_Kp) / (max_Kp - min_Kp) * (2**bits - 1)), f'0{bits}b')
    Ki_bin = format(int((Ki - min_Ki) / (max_Ki - min_Ki) * (2**bits - 1)), f'0{bits}b')
    Kd_bin = format(int((Kd - min_Kd) / (max_Kd - min_Kd) * (2**bits - 1)), f'0{bits}b')
    return Kp_bin, Ki_bin, Kd_bin

def decode_pid(gene, Kp_range, Ki_range, Kd_range, bits=8):
    min_Kp, max_Kp = Kp_range
    min_Ki, max_Ki = Ki_range
    min_Kd, max_Kd = Kd_range
    Kp = min_Kp + (max_Kp - min_Kp) * int(gene[0:bits], 2) / (2**bits - 1)
    Ki = min_Ki + (max_Ki - min_Ki) * int(gene[bits:2*bits], 2) / (2**bits - 1)
    Kd = min_Kd + (max_Kd - min_Kd) * int(gene[2*bits:3*bits], 2) / (2**bits - 1)
    return Kp, Ki, Kd

def evaluate_pid(population):
    results = []
    for ind in population:
        Kp, Ki, Kd = decode_pid(ind, Kp_range, Ki_range, Kd_range)
        t, y, error = simulate_step_response(Kp, Ki, Kd)
        iae = f_IAE(error, t)
        itae = f_ITAE(error, t)
        mse = f_MSE(error, t)
        results.append((iae, itae, mse))
    return results
def errors_to_fitness(results):
    fitness_IAE  = [1 / (1 + r[0]) for r in results]
    fitness_ITAE = [1 / (1 + r[1]) for r in results]
    fitness_MSE  = [1 / (1 + r[2]) for r in results]
    return fitness_IAE, fitness_ITAE, fitness_MSE

def roulette_selection(population, fitnesses):
    selected = []
    total_fitness = sum(fitnesses)
    probs = [fit / total_fitness for fit in fitnesses]
    r = random.random()
    cumulative = 0
    for ind, prob in zip(population, probs):
        cumulative += prob
        if r <= cumulative:
            selected.append(ind)
            break
    else:
        selected.append(population[-1])
    return selected

def crossover(parent1, parent2, blockSize=8, Pc=0.8):
    child1, child2 = "", ""
    r = random.random()
    for i in range(0, len(parent1), blockSize):
        block1 = parent1[i:i+blockSize]
        block2 = parent2[i:i+blockSize]
        if r < Pc:
            point = random.randint(1, blockSize - 1)
            c1 = block1[:point] + block2[point:]
            c2 = block2[:point] + block1[point:]
        else:
            c1, c2 = block1, block2
        child1 += c1
        child2 += c2
    return child1, child2

def mutate(individual):
    idx = list(individual)
    r = random.random()
    if r < Pm:
        point = random.randint(0, GENE_LENGTH - 1)
        idx[point] = '1' if idx[point] == '0' else '0'
    return ''.join(idx)

def elitism(population, fitnesses, n=1):
    selected = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [ind for ind, fit in selected[:n]]

if __name__ == "__main__":
    population = []
    population_IAE = []
    population_ITAE = []
    population_MSE = []
    best_IAE = []
    best_ITAE = []
    best_MSE = []
    for _ in range(populationSize):
        Kp = random.uniform(Kp_range[0], Kp_range[1])
        Ki = random.uniform(Ki_range[0], Ki_range[1])
        Kd = random.uniform(Kd_range[0], Kd_range[1])
        gene = ''.join(encode_pid(Kp, Ki, Kd, Kp_range, Ki_range, Kd_range))
        population.append(gene)
    population_IAE  = population[:]
    population_ITAE = population[:]
    population_MSE  = population[:]
    global_best_fitness_IAE  = -float("inf")
    global_best_gene_IAE     = None

    global_best_fitness_ITAE = -float("inf")    
    global_best_gene_ITAE    = None

    global_best_fitness_MSE  = -float("inf")
    global_best_gene_MSE     = None
    for generation in range(GENERATIONS):
        # --- IAE ---
        results_IAE = evaluate_pid(population_IAE)
        fitness_IAE,_,_ = errors_to_fitness(results_IAE)
        best_idx_IAE = np.argmax(fitness_IAE)
        best_fitness_IAE = fitness_IAE[best_idx_IAE]
        best_gene_IAE = population_IAE[best_idx_IAE]
        if best_fitness_IAE > global_best_fitness_IAE:
            global_best_fitness_IAE = best_fitness_IAE
            global_best_gene_IAE = best_gene_IAE
        best_IAE.append(global_best_fitness_IAE)

        # --- ITAE ---
        results_ITAE = evaluate_pid(population_ITAE)
        _,fitness_ITAE,_ = errors_to_fitness(results_ITAE)
        best_idx_ITAE = np.argmax(fitness_ITAE)
        best_fitness_ITAE = fitness_ITAE[best_idx_ITAE]
        best_gene_ITAE = population_ITAE[best_idx_ITAE]
        if best_fitness_ITAE > global_best_fitness_ITAE:
            global_best_fitness_ITAE = best_fitness_ITAE
            global_best_gene_ITAE = best_gene_ITAE
        best_ITAE.append(global_best_fitness_ITAE)

        # --- MSE ---
        results_MSE = evaluate_pid(population_MSE)
        _,_,fitness_MSE = errors_to_fitness(results_MSE)
        best_idx_MSE = np.argmax(fitness_MSE)
        best_fitness_MSE = fitness_MSE[best_idx_MSE]
        best_gene_MSE = population_MSE[best_idx_MSE]
        if best_fitness_MSE > global_best_fitness_MSE:
            global_best_fitness_MSE = best_fitness_MSE
            global_best_gene_MSE = best_gene_MSE
        best_MSE.append(global_best_fitness_MSE)
        elite_IAE = elitism(population_IAE, fitness_IAE, n=1)
        new_population_IAE = elite_IAE[:]
        while len(new_population_IAE) < populationSize:
            parent1 = roulette_selection(population_IAE, fitness_IAE)[0]
            parent2 = roulette_selection(population_IAE, fitness_IAE)[0]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population_IAE.extend([child1, child2])
        population_IAE = new_population_IAE[:]

        elite_ITAE = elitism(population_ITAE, fitness_ITAE, n=1)
        new_population_ITAE = elite_ITAE[:]
        while len(new_population_ITAE) < populationSize:
            parent1 = roulette_selection(population_ITAE, fitness_ITAE)[0]
            parent2 = roulette_selection(population_ITAE, fitness_ITAE)[0]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population_ITAE.extend([child1, child2])
        population_ITAE = new_population_ITAE[:]

        elite_MSE = elitism(population_MSE, fitness_MSE, n=1)
        new_population_MSE = elite_MSE[:]
        while len(new_population_MSE) < populationSize:
            parent1 = roulette_selection(population_MSE, fitness_MSE)[0]
            parent2 = roulette_selection(population_MSE, fitness_MSE)[0]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population_MSE.extend([child1, child2])
        population_MSE = new_population_MSE[:]
        population_IAE  = population_IAE
        population_ITAE = population_ITAE
        population_MSE  = population_MSE
    best_idx_IAE = fitness_IAE.index(max(fitness_IAE))
    best_gene_IAE = population_IAE[best_idx_IAE]
    best_idx_ITAE = fitness_ITAE.index(max(fitness_ITAE))
    best_gene_ITAE = population_ITAE[best_idx_ITAE]
    best_idx_MSE = fitness_MSE.index(max(fitness_MSE))
    best_gene_MSE = population_MSE[best_idx_MSE]
    print("=== IAE Optimization ===")
    print(f"Best IAE Fitness: {global_best_fitness_IAE}")
    print(f"Best IAE Gene: {global_best_gene_IAE}")
    print("=== ITAE Optimization ===")
    print(f"Best ITAE Fitness: {global_best_fitness_ITAE}")
    print(f"Best ITAE Gene: {global_best_gene_ITAE}")
    print("=== MSE Optimization ===")
    print(f"Best MSE Fitness: {global_best_fitness_MSE}")
    print(f"Best MSE Gene: {global_best_gene_MSE}")

    Kp_IAE, Ki_IAE, Kd_IAE = decode_pid(best_gene_IAE, Kp_range, Ki_range, Kd_range)
    print(f"Best PID parameters IAE: Kp={Kp_IAE}, Ki={Ki_IAE}, Kd={Kd_IAE}")
    Kp_ITAE, Ki_ITAE, Kd_ITAE = decode_pid(best_gene_ITAE, Kp_range, Ki_range, Kd_range)
    print(f"Best PID parameters ITAE: Kp={Kp_ITAE}, Ki={Ki_ITAE}, Kd={Kd_ITAE}")
    Kp_MSE, Ki_MSE, Kd_MSE = decode_pid(best_gene_MSE, Kp_range, Ki_range, Kd_range)
    print(f"Best PID parameters MSE: Kp={Kp_MSE}, Ki={Ki_MSE}, Kd={Kd_MSE}")
    t_IAE, y_IAE, error_IAE = simulate_step_response(Kp_IAE, Ki_IAE, Kd_IAE)
    t_ITAE, y_ITAE, error_ITAE = simulate_step_response(Kp_ITAE, Ki_ITAE, Kd_ITAE)
    t_MSE, y_MSE, error_MSE = simulate_step_response(Kp_MSE, Ki_MSE, Kd_MSE)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_IAE, y_IAE, label='GA-IAE')
    plt.title('Step Response with GA-IAE Optimized PID')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t_ITAE, y_ITAE, label='GA-ITAE', color='orange')
    plt.title('Step Response with GA-ITAE Optimized PID')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t_MSE, y_MSE, label='GA-MSE', color='green')
    plt.title('Step Response with GA-MSE Optimized PID')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(t_IAE, error_IAE, label="IAE Error")
    plt.plot(t_ITAE, error_ITAE, label="ITAE Error")
    plt.plot(t_MSE, error_MSE, label="MSE Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title("Error Signal for Optimized PID Controllers")
    plt.legend()
    plt.grid(True)
    plt.show()

    t_IAE1, y_IAE1, error_IAE1 = simulate_step_response(Kp_ITAE, Ki_ITAE, Kd_ITAE)
    t_IAE2, y_IAE2, error_IAE2 = simulate_step_response(Kp_MSE, Ki_MSE, Kd_MSE)
    t_ITAE1, y_ITAE1, error_ITAE1 = simulate_step_response(Kp_IAE, Ki_IAE, Kd_IAE)
    t_ITAE2, y_ITAE2, error_ITAE2 = simulate_step_response(Kp_MSE, Ki_MSE, Kd_MSE)
    t_MSE1, y_MSE1, error_MSE1 = simulate_step_response(Kp_IAE, Ki_IAE, Kd_IAE)
    t_MSE2, y_MSE2, error_MSE2 = simulate_step_response(Kp_ITAE, Ki_ITAE, Kd_ITAE)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_IAE1, y_IAE1, label='GA-IAE with ITAE PID', linestyle='--')
    plt.plot(t_IAE2, y_IAE2, label='GA-IAE with MSE PID', linestyle=':')
    plt.title('Step Response with GA-IAE PID on Other Criteria')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t_ITAE1, y_ITAE1, label='GA-ITAE with IAE PID', linestyle='--', color='orange')
    plt.plot(t_ITAE2, y_ITAE2, label='GA-ITAE with MSE PID', linestyle=':', color='red')
    plt.title('Step Response with GA-ITAE PID on Other Criteria')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t_MSE1, y_MSE1, label='GA-MSE with IAE PID', linestyle='--', color='green')
    plt.plot(t_MSE2, y_MSE2, label='GA-MSE with ITAE PID', linestyle=':', color='lime')
    plt.title('Step Response with GA-MSE PID on Other Criteria')
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

