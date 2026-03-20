import matplotlib.pyplot as plt

dataset = [2000, 10000, 1000000]
loss = [5.12, 4.85, 4.21]

plt.figure(figsize=(6,4))
plt.plot(dataset, loss, marker='o')

plt.xscale("log")
plt.xlabel("Dataset Size")
plt.ylabel("Training Loss")
plt.title("Scaling Behavior: Dataset Size vs Loss")

plt.grid(True)

plt.savefig("scaling_plot.png", dpi=300)