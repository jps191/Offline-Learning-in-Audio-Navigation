#NOTE parts of this project were too large even after compression, specfically the replay buffer, run logs, and unity envionment


#TD3 audio navigation project
This project aims to test how a replay buffer can influence an agents ability to naviagte
an audio environment

#setting up the environment
first you need to pull the oringinal repo from this link
https://github.com/petrosgk/AudioRL
swap the asset folder in this project with the one present in this repo

Since the original project was made in 2017, you will need to download unity 2021.2.45f2
#once you have it installed you will need to change the path set in mainfest.json
"com.unity.ai.navigation":"file:Packages/com.unity.ai.navigation", to the proper path in your PC 

Next you need to create a virtual environment containing the dependencies for ml agents show below
mlagents==0.30.0
├── grpcio [required: >=1.11.0, installed: 1.76.0]
│   └── typing_extensions [required: ~=4.12, installed: 4.15.0]
├── h5py [required: >=2.9.0, installed: 3.14.0]
│   └── numpy [required: >=1.19.3, installed: 1.21.2]
├── mlagents_envs [required: ==0.30.0, installed: 0.30.0]
│   ├── cloudpickle [required: Any, installed: 3.1.2]
│   ├── grpcio [required: >=1.11.0, installed: 1.76.0]
│   │   └── typing_extensions [required: ~=4.12, installed: 4.15.0]
│   ├── numpy [required: >=1.14.1, installed: 1.21.2]
│   ├── pillow [required: >=4.2.1, installed: 11.3.0]
│   ├── protobuf [required: >=3.6, installed: 3.20.0]
│   ├── PyYAML [required: >=3.1.0, installed: 6.0.3]
│   ├── gym [required: >=0.21.0, installed: 0.26.2]
│   │   ├── numpy [required: >=1.18.0, installed: 1.21.2]
│   │   ├── cloudpickle [required: >=1.2.0, installed: 3.1.2]
│   │   ├── importlib_metadata [required: >=4.8.0, installed: 8.7.0]
│   │   │   └── zipp [required: >=3.20, installed: 3.23.0]
│   │   └── gym-notices [required: >=0.0.4, installed: 0.1.0]
│   ├── PettingZoo [required: ==1.15.0, installed: 1.15.0]
│   │   ├── numpy [required: >=1.18.0, installed: 1.21.2]
│   │   └── gym [required: >=0.21.0, installed: 0.26.2]
│   │       ├── numpy [required: >=1.18.0, installed: 1.21.2]
│   │       ├── cloudpickle [required: >=1.2.0, installed: 3.1.2]
│   │       ├── importlib_metadata [required: >=4.8.0, installed: 8.7.0]
│   │       │   └── zipp [required: >=3.20, installed: 3.23.0]
│   │       └── gym-notices [required: >=0.0.4, installed: 0.1.0]
│   ├── numpy [required: ==1.21.2, installed: 1.21.2]
│   └── filelock [required: >=3.4.0, installed: 3.19.1]
├── numpy [required: >=1.13.3,<2.0, installed: 1.21.2]
├── pillow [required: >=4.2.1, installed: 11.3.0]
├── protobuf [required: >=3.6, installed: 3.20.0]
├── PyYAML [required: >=3.1.0, installed: 6.0.3]
├── tensorboard [required: >=1.15, installed: 2.20.0]
│   ├── absl-py [required: >=0.4, installed: 2.3.1]
│   ├── grpcio [required: >=1.48.2, installed: 1.76.0]
│   │   └── typing_extensions [required: ~=4.12, installed: 4.15.0]
│   ├── Markdown [required: >=2.6.8, installed: 3.9]
│   │   └── importlib_metadata [required: >=4.4, installed: 8.7.0]
│   │       └── zipp [required: >=3.20, installed: 3.23.0]
│   ├── numpy [required: >=1.12.0, installed: 1.21.2]
│   ├── packaging [required: Any, installed: 25.0]
│   ├── pillow [required: Any, installed: 11.3.0]
│   ├── protobuf [required: >=3.19.6,!=4.24.0, installed: 3.20.0]
│   ├── setuptools [required: >=41.0.0, installed: 58.1.0]
│   ├── tensorboard-data-server [required: >=0.7.0,<0.8.0, installed: 0.7.2]
│   └── Werkzeug [required: >=1.0.1, installed: 3.1.4]
│       └── MarkupSafe [required: >=2.1.1, installed: 2.1.5]
├── cattrs [required: >=1.1.0,<1.7, installed: 1.5.0]
│   └── attrs [required: >=20.1.0, installed: 25.4.0]
├── attrs [required: >=19.3.0, installed: 25.4.0]
└── pypiwin32 [required: ==223, installed: 223]
    └── pywin32 [required: >=223, installed: 311]
    onnx==1.12.0
├── numpy [required: >=1.16.6, installed: 1.21.2]
├── protobuf [required: >=3.12.2,<=3.20.1, installed: 3.20.0]
└── typing_extensions [required: >=3.6.2.1, installed: 4.15.0]

# to train run
mlagents-learn configuration.yaml --run-id=AudioRL_RunX

#to test run with preferred model in unity environment with train set to off
once 100 epsidoes pass it will display the success rate of the model

#setting up the td3 training model
you must add the td3 folder to the mlagent site package to be able to reference it specifically lib > site-packages > mlagents > trainers

