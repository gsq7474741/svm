import nni
from nni.experiment import Experiment, RemoteMachineConfig

if __name__ == '__main__':
    search_space = {
        'c': {'_type': 'uniform', '_value': [0, 20]},
        'gamma': {'_type': 'loguniform', '_value': [1e-10, 10]},
    }
    experiment = Experiment('local')
    # experiment = Experiment('remote')
    # mac_config = RemoteMachineConfig(host='192.168.0.100', user='songqi', password='Aa7474741',
    #                                  python_path='/Users/songqi/anaconda3/envs/svm/bin/python')
    # experiment.config.machines.append(mac_config)
    # experiment.config.nni_manager_ip
    experiment.config.trial_command = 'python train_np.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 512
    experiment.config.trial_concurrency = 16
    experiment.run(8001)
    input('Press enter to quit')
    experiment.stop()

    # experiment.config.trial_command = 'python3 trial.py'

    # host: str
    # port: int = 22
    # user: str
    # password: Optional[str] = None
    # ssh_key_file: Optional[utils.PathLike] = '~/.ssh/id_rsa'
    # ssh_passphrase: Optional[str] = None
    # use_active_gpu: bool = False
    # max_trial_number_per_gpu: int = 1
    # gpu_indices: Union[List[int], int, str, None] = None
    # python_path: Optional[str] = None


    # searchSpaceFile: search_space.json
    # trialCommand: python3 mnist.py
    # trialGpuNumber: 0
    # trialConcurrency: 4
    # maxTrialNumber: 20
    # tuner:
    # name: TPE
    # classArgs:
    # optimize_mode: maximize
    # trainingService:
    # platform: remote
    # machineList:
    # - host: 192.0.2.1
    # user: alice
    # ssh_key_file: ~/.ssh/id_rsa
    # - host: 192.0.2.2
    # port: 10022
    # user: bob
    # password: bob123
