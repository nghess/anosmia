import torch
import torch.multiprocessing as mp
import time
import os




def run_with_device(args):
    """
    Function to run some computation on a specific MIG device
    """

    mig_device, i, n, data = args
    print(f"Starting {i}")

    # Set the envirnment variable so that this process only sees the assigned MIG device
    os.environ['CUDA_VISIBLE_DEVICES'] = mig_device

    # Initialize the device and move the data to the device
    device = torch.device('cuda:0')
    data = data.to(device)

    # Perform some computation
    for _ in range(n):
        results = torch.matmul(data, data)

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
    data = torch.randn(100, 100, 100).to('cpu')
    n = 1000  # Number of iterations to simulate some computation


    # running sequenctially
    start_time = time.time()
    for i, mig_device in enumerate(mig_devices):
        args = (mig_device, i, n, data)
        run_with_device(args)
    print(f"\nTotal time: {time.time() - start_time} seconds")



    # Testing the parallel processing across the MIG devices
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
    start_time = time.time()
    args_list = [(mig_devices[i], i, n, data) for i in range(len(mig_devices))]
    with mp.Pool(len(mig_devices)) as pool:
        results = pool.map(run_with_device, args_list)
    print(f"Total time: {time.time() - start_time} seconds")

    



