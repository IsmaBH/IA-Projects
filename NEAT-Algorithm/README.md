## NEUROEVOLUTION OF AUGMENTING TOPOLOGIES
In this series of files I will attemp to implement my own implementation of the NEAT algorithm, first I will describe the process
followed by genetic algorithms.\\
### Genetic Algorithms
The genetic algorithms follows 5 important step:
1) Initialize the population. \\
    a) This usually starts the population of individuals with random sets of genomes
2) Evaluate the individuals
    a) This is done with a function that outputs a number of how well the answers is, this function is called the **fitness function**
        and the output number is the **fitness** of the individual.
3) Select the N best individuals
    a) Only the best fitness are selected to produce the next generation of the population, often via crossover or mutation.
4) Produce the next generation via genetic operations(crossover and mutation)
    a) Crossover is when we combine the genomes if 2 individuals in hope to create a better offspring.
    b) Mutation is when we change randomly a gene or genes of the genome of the individual.
5) Repeat 2-4 until goal is achived
    a) Usually this occurs when the desired fitness is reached or the algorithm can't improve more.

Important terms for this algorithm: 
Genotoype
: This are the actual genes of an individual, for this case the information to build the neural network is the genotype of the individuals.
Phenotype
: This are the visible traits of the individual, for this case the neural network is the phenotype of the individuals.

### Neural Networks
This are structures designed to mimic how the human brain learns by abstracting its functions with math, the most basic unit is the neuron. \\
The neurons for this case will have an **activation function**, a **bias** and an **id**, this last one is important as in the future, to be able to do the crossover and mutation operations an id will be necessary.\\
Then to be able to build a **network** we will need more than one single neuron, this neurons will be connected by the **synapses** or links which will have **register** of the neurons connected by it, a **weight** which represents the intensity of the connection and for this case a boolean value to indicate if the connection is enable.