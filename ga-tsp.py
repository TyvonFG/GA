import random
import math
import matplotlib.pyplot as plt #  pip install matplotlib
from statsmodels.stats.weightstats import ttest_ind #  pip install statsmodels
import statsmodels.stats.api as sms
import statistics


# Author: Tyvon Factor-Gaymon           Student ID #: 6580310
# Euclidean Distance Formula obtained from: https://www.cuemath.com/euclidean-distance-formula/

'''
The Changeable PARAMETERS underneath are the ones I used to modify the experiements. the only thing that cannot be modified here is the random seed
which is incremented in a for loop start from 1-5 for 5 sifferent seeds.
'''
# PARAMETERS
POPULATION_SIZE = 100
CROSSOVER_RATE =  0.8                                                                                                                                                                                                                                                                                                                                                
MUTATION_RATE =   0.2
SELECTION_K = 2
ELITISM_RATE = 0.1
MAX_GENERATION_SPAN = 100
MAX_RUNS = 5 
ORDERED = False # If set to False GA will perform Uniform Crossover instead
GRAPH = True # If set to False, no graph will be created
# Random Seed parameter is in run_experiment Function

# Non-changeable PARAMETERS
city_dict = {} # Dict of cities with their locations
chromosomes = {} # List of permuted cities / Initial Population

'''
* This function is used to process the city data from a textfile.
* It does this by making a dictionary that stores the chromosome and its x and y values.
* It then permutes the data so that we have a variety of different chromosomes.
* These chromosomes are then evaluated and given a Total Distance (Fitness) Value.
'''
def process_city_data( ):
    city_list = [] 

    file = open("eil51.tsp.txt","r")
    for line in file:
        if line == "\n":
            print(f'\nFinished reading File.\n')
            break
        if line[1].isdigit():   #Start reading when we encounter numbers
            city_list.append(line[:-1]) 
    file.close()  
    create_city_dict(city_list)
    permute_cities() # Takes the original list of cities and permutes them / Original Population
    initial_evaluation() # Intialize original population with fitnessoriginal_population


'''
* This function takes in a list of cities and their x and y coordinates
* and creates a city dictionary out of them. 
* Each Key in the dict is represented by a city with a unique x and y value.

'''
def create_city_dict(list):
    for city in list:
        city = city.split()
        city[1] = float(city[1]) # convert str to float
        city[2] = float(city[2]) # convert str to float   
        city_dict[city[0]]= {'x': city[1], 'y': city[2]}
    
'''
* This function joins together all the cities that were in the city dictionary
* and creates a list out of them.
* This list is then subjected to random permutations so that we can get different
* unique chromosomes.
'''
def permute_cities():
    temp_list = []
    permutation = ''
    for key in city_dict: # joins together the keys (cities) that are in a dict
        temp_list.append(key)

    for i in range (POPULATION_SIZE):
        #permutation = list(np.random.permutation(temp_list))
        permutation = random.sample(temp_list, k= len(temp_list))
        chromosomes[i] = {'chromosome' :permutation, 'total_distance': None}

'''
* This function is used to evaluate the intial population of chromosomes
* and initialize their fitness.
'''
def initial_evaluation():
    counter = 0
    for i in range (len(chromosomes)):
        chromosomes[i]['total_distance'] = evaluate(chromosomes[i]['chromosome'])
    
'''
* This function is used to evaluate a chromosome of chromosomes
* and initialize their fitness.
* It takes the sum of the Euclidean distances and uses that as their fitness values.
* return -- total_distance -- Fitness Value
'''
def evaluate(a_chromosome):
    total_distance = 0
    #distance = []

    for i in range (len(a_chromosome)-1):
        total_distance += calc_euclidean_distance(a_chromosome[i], a_chromosome[i+1])
        #distance.append(calc_euclidean_distance(a_chromosome[i], a_chromosome[i+1])) Distance between 2 cities
    return total_distance


'''
* This function takes two cities and calculated the Euclidean distance between them.
* It then returns this eculidean distance.
* return -- distance -- euclidean distance 
'''
def calc_euclidean_distance(city_1, city_2):
    x1,x2 = city_dict[city_1]['x'], city_dict[city_2]['x']
    y1, y2 = city_dict[city_1]['y'], city_dict[city_2]['y']
    
    distance =  math.sqrt((x2-x1)**2 +(y2-y1)**2) 
    return distance

'''
* This function takes in an initial population of chromosomes and subjects them
* to a Tournament K Selection. 
* The Chromosomes that win during this selection will be paired up in order
* and these pairs will be used as the mating pool.
* return -- selected_pairs -- This is a dictionary of selected pairs (paired up parents)
'''
def tournament_selection(initial_population):
    selected_pairs = {}
    selected = {}
    if POPULATION_SIZE%2 == 0: # POP_SIZE is even
        population = POPULATION_SIZE
    else:
        population = POPULATION_SIZE+1 # This allows us to have enough children. We simply delete 1 child later on

    for i in range(population//2): # Divide it by 2 because inner loop performs Torunament twice
        fittest = {}
        for k in range (2): # gets us a pair of paretns to mate
            for j in range(SELECTION_K):
                selected[j] = initial_population[random.randint(0,POPULATION_SIZE-1)]
            fittest[k] = compare(selected) # the chromosome with the shortest distance
        selected_pairs[i] = fittest
    return selected_pairs

'''
* This function takes in multiple chromosomes and compares 
* their fitness (total_distance) values between them.
* return -- fittest_chromosome -- The chromosome with the shorted total distance is returned
'''
def compare(selected_list):
    min_distance = float('inf')
    fittest_chromosome = {}
    for i in range (len(selected_list)):
        if min_distance > selected_list[i]['total_distance']:
            min_distance = selected_list[i]['total_distance']
            fittest_chromosome = selected_list[i]
    return fittest_chromosome
    
'''
* This function takes in a parent dictionary with the paired parents and subjects
* them to a crossover if R < CROSSOVER_RATE
* If parents do undergo crossover then their children will replace them within the new population (dictionary)
* Children are stored in the dict like their parents, in pairs
* If the parents do not undergo crossover then they will remain in the new population (dictionary)
* In this way, the parents become the new children.
* var -- Pc -- Probability of Crossover
* return -- parents -- a little misleading, this is simply the new population being returned
'''
def uniform_crossover(parents, Pc): # Uniform Order Crossover
    pair_updated = []
    for pair in range (len(parents)):
        parent_1 = parents[pair][0]['chromosome'].copy()
        parent_2 = parents[pair][1]['chromosome'].copy()

        R = random.random() # Randomly chosen number
        if R > Pc:
            continue # If Pc < R then Do not apply crossover and skip 

        pair_updated += [pair]
        mask = [random.randint(0,1) for _ in range(len(city_dict))]
        child_1 = []
        child_2 = []
        #p1 = chromosome_1['chromosome'].copy() # Elements of c1 when mask is 0
        #p2 = chromosome_2['chromosome'].copy() # Elements of c2 when mask is 0
        for c1,c2,mask in zip(parent_1, parent_2, mask):
            if mask == 1:
                child_1.append(c1)
                child_2.append(c2)
            else:
                child_1.append(None)
                child_2.append(None)

        for i in range(len(child_1)):
            if child_1[i] == None:
                for x in parent_2:
                    if x not in child_1:
                        child_1[i]=x
                        parent_2.remove(x)
                        break
            if child_2[i] == None:
                for z in parent_1:
                    if z not in child_2:
                        child_2[i] = z    
                        parent_1.remove(z)
                        break                    
        parents[pair][0]= {'chromosome': child_1, 'total_distance': evaluate(child_1)} 
        parents[pair][1] = {'chromosome': child_2, 'total_distance': evaluate(child_2)}
        #print()
    #print(pair_updated)
    return parents # This is the new population being returned (All children)

'''
* This function takes in a parent dictionary with the paired parents and subjects
* them to a crossover if R < CROSSOVER_RATE
* If parents do undergo crossover then their children will replace them within the new population (dictionary)
* Children are stored in the dict like their parents, in pairs
* If the parents do not undergo crossover then they will remain in the new population (dictionary)
* In this way, the parents become the new children.
* var -- Pc -- Probability of Crossover
* return -- parents -- a little misleading, this is simply the new population being returned
'''

def ordered_crossover(parents, Pc): # Uniform Order Crossover

    #indices_updated = []
    for pair in range(len(parents)):

        R = random.random() # Randomly chocen number
        if R > Pc:
            continue  # If Pc < R then Do not apply crossover and skip 

        parent_1 = parents[pair][0]['chromosome'].copy()
        parent_2 = parents[pair][1]['chromosome'].copy()
        #indices_updated += [ch_1] + [ch_2]

        random_index_1 = random.randint(0, len(city_dict)-1)
        random_index_2 = random.randint(0, len(city_dict)-1)
        if random_index_1 < random_index_2:
            start = random_index_1
            end = random_index_2
        else:
            start = random_index_2
            end = random_index_1
        child_1 = [None] * len(parent_1)
        child_2 = [None] * len(parent_2)
   

        child_1[start:end+1] = parent_1[start:end+1] # Direct Inheritance between Parent 1 and Child 1
        child_2[start:end+1] = parent_2[start:end+1] # Direct Inheritance between Parent 2 and Child 2

        original_index = [] # a of genes not inherited in order after inheritance
        for i in (parent_1[end+1:]+parent_1[:start]): #starts from end of parent 
           original_index.append(parent_1.index(i))     
  
        for i in original_index:
            for x in parent_2:
                if x not in child_1:
                    child_1[i] = x
                    break
            for y in parent_1:
                if y not in child_2:
                    child_2[i] = y
                    break                

        parents[pair][0] = {'chromosome': child_1, 'total_distance': evaluate(child_1)} # Replace Parent_1
        parents[pair][1] = {'chromosome': child_2, 'total_distance': evaluate(child_2)} # Replace Parent_2
    #indices_updated.sort()
    return parents # misleading variable name -- actually new population, children

'''
* This function takes in the sibling pairs dictionary and separates them.
* Then the children chromosomes are subjected to mutation if R > MUTATION_RATE.
* This function will return the new population with the individuals that underwent a mutation.
* var -- Pm -- Probability of Mutation
'''

def mutate(siblings, Pm): #Inversion
    
    children  ={}
    j = 0
    for i in range (0, len(siblings)): # This loop is necessary to treat each sibling as an individual instead of a pair
        children[j] = siblings[i][0]       # Cleans up our dictionary and makes it reflect the pop size
        children[j+1] = siblings[i][1]
        j += 2
    
    for i in range (0, POPULATION_SIZE-1):
        R = random.random() # Randomly chosen number
        if R > Pm: 
            continue  # If Pc < R then Do not apply mutation
        child = children[i]['chromosome']
        #indices_updated += [i]    

        random_index_1 = random.randint(0, len(child)-1)
        random_index_2 = random.randint(0, len(child)-1)
        if random_index_1 < random_index_2:
            start = random_index_1
            end = random_index_2
        else:
            start = random_index_2
            end = random_index_1
        child[start:end+1] = reversed(child[start:end+1])

        children[i] = {'chromosome': child , 'total_distance': evaluate(child)}   
        #print(f'Mutated: {indices_updated}') 
    return children
'''
* This function takes in the current population and makes a list of the best chromosomes.
* The size of the list depends on the ELITISM_RATE
* return -- elites -- Only the chromosomse with the best fitness will be returned.
'''
def get_elites(current_population):
    temp_dict = current_population.copy()
    # here we sort a dictionary based on total distance and save them in elites
    elites = [(k, temp_dict[k]) for k in sorted(temp_dict, key=lambda x: temp_dict[x]['total_distance'], reverse=False)]
    transfer = int(len(elites)*ELITISM_RATE)
    return elites[:transfer]

'''
* This function takes in a list of elites and a dictionary comprised of the mutated
* population.
* Depending on the amount of elites we have we could have a population thats too big.
* Therefore, if population is odd we have to get rid of a child to ensure our 
* population_size doesn't change. We then replace the worst chromosomes with the elites
* If the population is even then we simply replace the worst chromosomes with the elites.
'''
def transfer_elites(elites, mutated_population):
    elites = dict(elites)

    if POPULATION_SIZE%2 == 0: # POP_SIZE is even
    # Sort list from worst to best solutions 
    # This way we can replace the worst solutions with the elites
        mutated_population = [(k, mutated_population[k]) for k in sorted(mutated_population, key=lambda x: mutated_population[x]['total_distance'], reverse=True)]
        mutated_population = dict(mutated_population)
        
        for child, elite in zip(mutated_population, elites):
            mutated_population[child] = elites[elite]
            
    else:
    # Sort list from best to worst solution so we can pop worst solution at end since we have more children then needed
        mutated_population = [(k, mutated_population[k]) for k in sorted(mutated_population, key=lambda x: mutated_population[x]['total_distance'], reverse=False)]
        worst_chromosomes = [x[0] for x in reversed(mutated_population)] # Worst chromosomes come first in list

        worst_chromosome = mutated_population[-1][0]    # retain worst chromosome key number to replace it
        if worst_chromosome != len(mutated_population)-1:
            mutated_population.pop() # remove last element
            mutated_population = sorted(list(mutated_population))  # replace worst chrom number with last chrome number
            mutated_population[-1] = list(mutated_population[-1])   # to maintain unique keys up to POP_size -1
            mutated_population[-1][0] = worst_chromosome
        else:
            mutated_population.pop() # pop last element and we don't have to worry about replacing its key
        
        mutated_population = dict(mutated_population)
        for child, elite in zip(worst_chromosomes, elites):
            mutated_population[child] = elites[elite]

    mutated_population = [(k, mutated_population[k]) for k in sorted(mutated_population, key=lambda x: mutated_population[x]['total_distance'], reverse=False)]
    mutated_population = dict(mutated_population)
    return dict(mutated_population)

'''
This function takes in the new population and computes the average distance/fitness value in the population and also gets the fittest chromosome.
'''
def compute_fitness_data(new_population):
    # This variable stores the average population size by summing up all total distances and diving it by the POP size
    average_distance = sum([x[1]['total_distance'] for x in list(new_population.items())])/len(new_population)
    fittest = [(k, new_population[k]) for k in sorted(new_population, key=lambda x: new_population[x]['total_distance'], reverse=False)] # sort list based off of total distance to get best chromosome
    fittest_chrom = fittest[0][0]
    fittest = new_population[fittest_chrom]
    return fittest, average_distance

'''
* This function takes in the average population fitness and the average fittest chromosomes and plots them on a graph.
* variable -- x -- Epoch, the number of generations
* variable -- y1 -- The average population fitness
* variable -- y2 -- The average Fittest chromosomes
'''
def graph_average_data (x, y1, y2):
    x = [g for g in range(MAX_GENERATION_SPAN)]
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.title("Experiment 5B: Fitness Value Curves for the Average Population fitness and Average Fittest individuals")
    plt.ylabel('Fitness Value')
    plt.xlabel('Generation')
    for var in (y1, y2):
        plt.annotate('%0.2f' % min(var), xy=(1, min(var)), xytext=(8, 0), 
            xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.legend(['Avg Pop Fitness','Average Fittest Individuals'])
    plt.show()
    print()

'''
* This function takes in a list of fitnesses whether it be the average population fitness or
* the average fittest individuals and takes an average of the averages.
* If we have 5 runs then we have 5 lists of averages for average population average and we take the average
* between all 5 lists since they were produced during different runs to get an average of the runs.
'''
def compute_averages(averages):
    avg_list = []
    
    for i in range(MAX_GENERATION_SPAN):
        generation = []
        for j in range (MAX_RUNS):
            generation.append(averages[j][i])
        avg_list.append(sum(generation)/MAX_RUNS)
    return avg_list
'''
* This function is used to evolve solutions from an intial permuted population of chromosomes (list or dict of cities).
* First the elites, the fittest chromosomes are chosen and will be added to the new population later.
* It subjects the initial population to a K Tournament to get parents that will be paired up and serve as the mating pool.
* These parents are then subjected to a crossover in which the children will make up the new population. 
* Parents become children if they aren't chosen for crossover. Children are stored as paris, like their parents.
* These children chromosomes are then subjected to mutation, forming the new population.
* The elites then get added to the new population, by replacing the chromosomes with the worst fitness values.
'''
def run_genetic_algorithm(Pc, Pm):
    original_population = chromosomes.copy() # Initial Population
    current_population = original_population # 
    fittest_chromosome = {}
    average_pop_fitness = [] #a list to store the average population fitness values from the generations
    fittest = [] #a list to store the 
 
    for g in range (MAX_GENERATION_SPAN):
    #elites
        elites = get_elites(current_population)
        parents = tournament_selection(current_population)  # Selected_population 
        if ORDERED:
            children = ordered_crossover(parents,Pc)  # Ordered_Crossover
        else:
            children = uniform_crossover(parents,Pc)  # Uniform_Crossover
         

        mutated_population = mutate(children,Pm) # Inversion Mutation
        new_population = transfer_elites(elites, mutated_population)
        current_population = new_population
        fittest_chromosome, average_distance = compute_fitness_data(new_population)
        average_pop_fitness.append(average_distance)
        fittest.append(fittest_chromosome['total_distance'])


        print(f'Generation: {g+1}\n')
        print(f'Average Populaton fitness Value: {average_distance}')
        print(f'Fittest Chromosome in Generation')
        print('Chromosome:' + str(fittest_chromosome['chromosome']))
        print('Total Distance:' + str(fittest_chromosome['total_distance'])+'\n')
    return  average_pop_fitness, fittest

'''
* This function is used to run a whole experiment. This means that there are a total of 5 runs being excecuted.
* This allows us to collect data at a much greater speed, and we can use the data to calculate averages.
* It takes in two variables which default to preset parameters if nothing is used as an arguement
* var -- Pc -- Probability of Crossover
* var -- Mc -- Probability of Mutation
* A random seed seeds the random numbers starting at 1-5, so we have a different seed per run.
* I hope you enjoyed the experiment! ^_^
'''

def run_experiment(Pc = CROSSOVER_RATE, Pm = MUTATION_RATE):
    average_pop_fittness = { }
    fittest = {}
    for i in range (MAX_RUNS):
        random.seed(i+1)   # PARAMETER
        process_city_data()
        average_pop_fittness[i], fittest[i] = run_genetic_algorithm(Pc, Pm)
    average_pop_fittness = compute_averages(average_pop_fittness)
    fittest = compute_averages(fittest)
    if GRAPH:
        graph_average_data(MAX_GENERATION_SPAN, average_pop_fittness, fittest)
    print("*******************************************************************************************************************")
    print('PARAMETERS')
    print(f'POPULATION_SIZE: {POPULATION_SIZE},    CROSSOVER_RATE: {Pc},   MUTATION_RATE:   {Pm}\n')     
    print(f'SELECTION_K: {SELECTION_K},   ELITISM_RATE: {ELITISM_RATE},   MAX_GENERATION_SPAN: {MAX_GENERATION_SPAN}\n')    
    print(f'Random Seeds: 1-{MAX_RUNS} in increasing order per run (different seed per Run)\n')       
    calculate_stats(average_pop_fittness, fittest)        
    print("*******************************************************************************************************************")
    print()
    return average_pop_fittness

'''
 * This function is responsible running two experiments and comparing the results using a ttest.
 * var -- samples --  is alist that holds the average population fitness of a whole experiment.
'''
def perform_ttest():
    samples = []
    for i in range(2):
        if i<1:
            samples.append(run_experiment())
        else: 
            samples.append(run_experiment(0.8,0.2)) # Change Pc and Mc here to compare this experiment with the first experiment with the orignial parameters
    ttest = ttest_ind(samples[0],samples[1])
    print(f'\n{ttest}\n')

'''
* This function is designed to provide us with the stats of both the average population fitness and average fittest chromosomes.
'''
def calculate_stats(avg_pop_fitness, fittest_chromosomes):
    mean_1 = statistics.mean(avg_pop_fitness)
    median_1 = statistics.median(avg_pop_fitness)
    sd_1 = statistics.stdev(avg_pop_fitness)
    min_val_1 = min(avg_pop_fitness)
    max_val_1 = max(avg_pop_fitness)

    mean_2 = statistics.mean(fittest_chromosomes)
    median_2 = statistics.median(fittest_chromosomes)
    sd_2 = statistics.stdev(fittest_chromosomes)
    min_val_2 = min(fittest_chromosomes)
    max_val_2= max(fittest_chromosomes)
    print(f'STATISTICS')
    print(f'Average Population Fitness Data:')
    print(f'Mean: {mean_1}   Median: {median_1},  Standard Deviation: {sd_1}   Min: {min_val_1}   Max: {max_val_1}\n')
    print(f'Fittest_Chromosomes Data:')
    print(f'Mean: {mean_2}   Median: {median_2},  Standard Deviation: {sd_2}   Min: { min_val_2}   Max: {max_val_2}')





#perform_ttest()
run_experiment()




#   Permutation
#   Evaluation
#       Elitism
#       Tournament K
#       Crossover
#       Mutation
#       
#
#
#
#
#