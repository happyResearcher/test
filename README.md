# Running a pytorch program with CPU/ 1 GPU/ Multi-GPU.
We give an example to show how to run a Python program with CPU, 1 GPU and multi-GPU.
The codes used in this example is modified from:
•	Tutorials > Deep Learning with PyTorch: A 60 Minute Blitz > Training a Classifier
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
•	Tutorials > Optional: Data Parallelism https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

1.Run the CPU version codes --- cifar10_tutorial_CPU.py

2.Run the 1GPU version codes --- cifar10_tutorial_1GPU.py
Compared with the CPU version, the 1GPU version has the following changes.
① we add the following lines in line 73-75.
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
print(device)
net.to(device) # transport the net to GPU

② we add the following lines in line 104-106.
 
% transport data to GPU
inputs=inputs.to(device)
labels=labels.to(device)

3.Run the multi-GPU version codes --- cifar10_tutorial_multi_gpu.py 
The only difference is that the following 2 lines are added in line 76-77, compared with the 1GPU version. 
if torch.cuda.device_count() > 1:
  net = nn.DataParallel(net)

Why don’t notice MASSIVE speedup compared to CPU? Because the network is really small.
