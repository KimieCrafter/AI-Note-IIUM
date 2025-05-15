import subprocess

# Run the nvidia-smi command and print its output
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout)
