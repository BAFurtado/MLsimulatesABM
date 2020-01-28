# Machine Learning simulates Agent-based Model

This is an initial exploratory exercise done for the class @ http://thiagomarzagao.com/teaching/ipea/
Text available here: https://arxiv.org/abs/1712.04429v1

The program:

1. Reads output from an ABM model and its parameters' configuration
2. Creates a socioeconomic optimal output based on two ABM results of the modelers choice
3. Organizes the data as X and Y matrices
4. Trains some Machine Learning algorithms
5. Generates random configuration of parameters based on the mean and standard deviation of the original parameters
6. Apply the trained ML algorithms to the set of randomly generated data
7. Outputs the mean and values for the actual data, the randomly generated data and the optimal and non-optimal cases

The original database from which the 232 samples of the actual data is read is large (60.7 GB)
Thus, some pre-processed data for some pairs of optimal cases are also made available

# Running the program
`python main.py`

Output will be produced at the output folder
You may change the parameters for the targets at main.py
Or you may change the parameters of the ML in machines.py
Or the size of the sample at generating_random_conf.py
