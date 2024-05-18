import os
import subprocess

def get_free_gpu():
    try:
        # Get GPU status from nvidia-smi
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'])
        gpu_status = result.decode('utf-8').strip().split('\n')
        
        # Parse the GPU status
        gpu_status = [line.split(', ') for line in gpu_status]
        gpu_status = [(int(index), int(memory_used)) for index, memory_used in gpu_status]

        # Sort GPUs by memory used (ascending)
        gpu_status.sort(key=lambda x: x[1])

        # Select the GPU with the least memory used
        free_gpu = gpu_status[0][0]
        return free_gpu
    except Exception as e:
        print(f"Error in getting free GPU: {e}")
        return None

if __name__ == "__main__":
        free_gpu = get_free_gpu()
        if free_gpu is not None:
            print(f"Setting CUDA_VISIBLE_DEVICES to {free_gpu}")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(free_gpu)
        else:
            print("No GPU available or error occurred.")

        # Here you can add code to run your training script or import it
        # Example:
        # import ppo_train
