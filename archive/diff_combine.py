from pathlib import Path
import os
import pickle as pk
import matplotlib.pyplot as plt

result_path = Path(__file__).parent / 'Results_Combine'
os.makedirs(str(result_path), exist_ok=True)

save_path = Path(__file__).parent / 'Combined_Plots'
os.makedirs(str(save_path), exist_ok=True)

def main():
    results = {}
    # Load all the data
    result_folders = [x for x in result_path.iterdir() if x.is_dir()]
    for folder in result_folders:
        object_files = list(folder.glob('**/*.p'))
        loaded_files = {}
        for file in object_files:
            loaded_files[file.name[:-2]] = pk.load(open(file, 'rb'))
        results[folder.name] = loaded_files

    # Instantaneous received combined plot
    plt.figure()
    for exp_name, exp_data in results.items():
        plt.plot(exp_data['time_array'], exp_data['avg_hits_inst'], label=exp_name)
    plt.xlabel('Time (s)'), plt.ylabel('Expected Received Particle Count Per Second')
    plt.title('Expected # of Instantaneous received (S)-Mandelate molecules')
    plt.legend()
    plt.savefig(save_path / 'S_received_inst.png')
    # Instantaneous released combined plot
    plt.figure()
    for exp_name, exp_data in results.items():
        plt.plot(exp_data['time_array'], exp_data['S_released_instant'], label=exp_name)
    plt.xlabel('Time (s)'), plt.ylabel('Particle Count Per Second')
    plt.title('Instantaneous # of released (S)-Mandelate molecules')
    plt.legend()
    plt.savefig(save_path / 'S_released_inst.png')
    # Total released combined plot
    plt.figure()
    for exp_name, exp_data in results.items():
        plt.plot(exp_data['time_array'], exp_data['S_released_count'], label=exp_name)
    plt.xlabel('Time (s)'), plt.ylabel('Particle Count')
    plt.title('Total # of released (S)-Mandelate molecules')
    plt.legend()
    plt.savefig(save_path / 'S_released.png')


    


if __name__ == '__main__':
    main()