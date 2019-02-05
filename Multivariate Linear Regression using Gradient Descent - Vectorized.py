import numpy as np 

# Input shape of feature matrix
n = 5
m = 10

# Feature matrix generation and reshaping for intercept
X = np.random.randint(1,50,(m,n))
X = np.insert(X,0,1,axis=1)
y = np.random.randint(1,50,(m,1))

# Learning rate and Number of iterations
alpha = 0.01
n_iter = 500

# Initialize Parameter Vector
theta = np.zeros((n+1,1))

# Compute value of cost function
def cost_fn(X,y,theta):
	return (sum((1/(2*m))*(X.dot(theta) - y)**2))

# Run gradiet descent for convergence
def grad_desc(X,y,theta,alpha,n_iter):
	cost_values = []
	m = len(y)

	for i in range(1,n_iter+1):
		# Gradient Step
		theta = theta - (1/m)*((((X.dot(theta) - y).T).dot(X)).T)*alpha
		# Store value of cost function
		cost_values.append(cost_fn(X,y,theta))

	return theta,cost_values

