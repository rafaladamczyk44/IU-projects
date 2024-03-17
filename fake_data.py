import numpy as np

# Define the number of samples for the first column
num_samples = 20

# Generate random values for the first column within the range -4 to 4
column1 = np.random.randint(-4, 5, size=num_samples)
column2 = np.random.randint(-4, 5, size=num_samples)
column3 = np.random.randint(-4, 5, size=num_samples)
column4 = np.random.randint(-4, 5, size=num_samples)

# Generate random classes (0 or 1)
classes = np.random.randint(2, size=num_samples)

# Combine columns and classes into a single dataset
dataset = np.column_stack((column1, column2, column3, column4, classes))

# Save the dataset to a CSV file
np.savetxt("3. SVM/fake_dataset.csv", dataset, delimiter=",", fmt="%f", header="a,b,c,d,Classes")
