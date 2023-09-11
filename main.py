# Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Generate Equity Line Data
np.random.seed(1000)
time_period = 2500  # Number of time periods (e.g., days)
good_equity = np.cumsum(np.random.normal(loc=0.1, scale=0.2, size=time_period))  # 'Good' equity line
bad_equity = np.cumsum(np.random.normal(loc=0.1, scale=0.5, size=time_period))  # 'Bad' equity line
time_vector = np.arange(1, time_period + 1)  # Time vector

# Adjust Equity Lines to Start from 0
good_equity = good_equity - good_equity[0]
bad_equity = bad_equity - bad_equity[0]

# Plot Equity Lines
plt.figure(figsize=(14, 8))
plt.plot(time_vector, good_equity, label='Good Equity Line', color='green')
plt.plot(time_vector, bad_equity, label='Bad Equity Line', color='red')
plt.xlabel('Time (days)')
plt.ylabel('Equity Value')
plt.title('Equity Lines')
plt.legend()
plt.grid(True)
plt.show()

# Function to Calculate SEE for Different Blocks
def calculate_see_blocks(equity_line, num_blocks):
    block_size = len(equity_line) // num_blocks
    see_values = []
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_equity = equity_line[start_idx:end_idx]
        block_time = time_vector[start_idx:end_idx]
        model = LinearRegression()
        model.fit(block_time.reshape(-1, 1), block_equity)
        predictions = model.predict(block_time.reshape(-1, 1))
        residuals = block_equity - predictions
        see = np.std(residuals)
        see_values.append(see)
    return see_values

# Function to Calculate Consistency of Linearity (CL)
def calculate_cl(see_values):
    return np.std(see_values)

# Number of Blocks to Divide the Equity Line Into
num_blocks = 10

# Calculate SEE Values for Each Block
good_see_blocks = calculate_see_blocks(good_equity, num_blocks)
bad_see_blocks = calculate_see_blocks(bad_equity, num_blocks)

# Plot SEE Across Blocks
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.bar(np.arange(num_blocks) - 0.2, good_see_blocks, width=0.4, label='Good Equity Line SEE', color='green')
ax1.bar(np.arange(num_blocks) + 0.2, bad_see_blocks, width=0.4, label='Bad Equity Line SEE', color='red')
ax1.set_xlabel('Block Number')
ax1.set_ylabel('SEE')
ax1.set_title('SEE Across Different Blocks')
ax1.legend()
plt.show()

# Calculate CL for Both Equity Lines
good_cl = calculate_cl(good_see_blocks)
bad_cl = calculate_cl(bad_see_blocks)

# Display Results
print(f"Good Equity Line CL: {good_cl}")
print(f"Bad Equity Line CL: {bad_cl}")
