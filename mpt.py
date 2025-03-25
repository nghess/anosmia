import torch
import torch.multiprocessing as mp
import time
import os




def run_with_device(args):
    """
    Function to run some computation on a specific MIG device
    """

    mig_device, i, n, data = args
    print(f"\nStarting {i}")

    # Print the visible devices
    print(f"Process {i} sees the following devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Process {i} sees the following devices: {torch.cuda.device_count()}")

    # Set the envirnment variable so that this process only sees the assigned MIG device
    #os.environ['CUDA_VISIBLE_DEVICES'] = mig_device

    # Initialize the device and move the data to the device
    transfer_start = time.time()
    device = torch.device('cuda:0')
    data = data.to(device)
    print(f"Data transfer time for process {i}: {time.time() - transfer_start} seconds")

    # Perform some computation
    weights1 = torch.randn(100_000, 1024).to(device)
    weights2 = torch.randn(1024, 1024).to(device)
    for _ in range(n):
        intermediate = torch.matmul(data, weights1)
        output = torch.matmul(intermediate, weights2)

    print(f"Process {i} done")




if __name__ == "__main__":


    # Set the start method to spawn
    mp.set_start_method('spawn', force=True)


    # Check the number of available cuda devices (MIG instances) and collect their names.
    print(f'Devices recognized by PyTorch: {torch.cuda.device_count()}') #Note that PyTorch only recognizes a single device
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    mig_devices = []
    if devices:
        device_list = devices.split(',')
        print(f'\nNumber of visible cuda devices: {len(device_list)}')
        for i, d in enumerate(device_list):
            print(f"Device {i} name: {d}")
            mig_devices.append(d)
    else:
        print("No visible CUDA devices found.")


    # Make the data
    data = torch.randn(1024, 100_000).to('cpu')
    n = 100  # Number of iterations to simulate some computation

    # running sequenctially
    print("\n\nRunning sequentially")
    start_time = time.time()
    for i, mig_device in enumerate(mig_devices):
        args = (mig_device, i, n, data)
        run_with_device(args)
    print(f"\nTotal time: {time.time() - start_time} seconds")



    # Testing the parallel processing across the MIG devices
    print("\n\n\nRunning in parallel")
    start_time = time.time()
    processes = []
    for i, mig_device in enumerate(mig_devices):
        args = mig_device, i, n, data
        p = mp.Process(target=run_with_device, args=(args,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(f"Total time: {time.time() - start_time} seconds")



    # Testing the parallel processing across the MIG devices with Pool executer
    print("\n\nRunning in parallel with Pool executer")
    start_time = time.time()
    args_list = [(mig_devices[i], i, n, data) for i in range(len(mig_devices))]
    with mp.Pool(processes = len(mig_devices)) as pool:
        results = pool.map(run_with_device, args_list)
    print(f"Total time: {time.time() - start_time} seconds")

    



