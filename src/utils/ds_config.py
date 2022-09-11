import pandas as pd
def create_config_file(path):
    column =["DS_NAME", "MODALITY_NAME", "MODALITY_CH", "WINDOW", "CODE"]
    config = [["USCHAR", ["acc","gyro"],[3,3], 400,20],
              ["WISDM", ["acc", "gyro"], [3, 3], 200, 20],
              ["SLEEPEDF", ["EEG", "EOG"], [1, 1], 3000, 20],
              ["S-WISDM", ["acc", "gyro"], [3, 3], 200, 20]]
    df = pd.DataFrame(data= config, columns=column)

def read_config(ds_name):
    if ds_name == "OPP":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["ACC1","ACC2","ACC3","ACC4"],
            "MODALITY_CH": [3,3,3,3],
            "WINDOW": 60,
            "CODE": 20,
            "CLASS":4,
            "HWIN" :48,
            "FWIN" :12,
            "LABEL_NAME":["Sitting", "Standing", "Walking", "Laying"]
        }
    elif ds_name == "OPP2":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["ACC1","ACC2"],
            "MODALITY_CH": [3,3],
            "WINDOW": 60,
            "CODE": 20,
            "CLASS":4,
            "HWIN" :48,
            "FWIN" :12,
            "LABEL_NAME":["Sitting", "Standing", "Walking","Laying"]
        }
    elif ds_name == "USCHAR":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["acc","gyro"],
            "MODALITY_CH": [3,3],
            "WINDOW": 400,
            "CODE": 20,
            "CLASS":6,
            "HWIN" :300,
            "FWIN" :100,
            "LABEL_NAME": ["Walking","Walking Up","Walking Down","Sitting","Standing", "Laying"]
        }
    elif ds_name in ["WISDM","S-WISDM"]:
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["acc","gyro"],
            "MODALITY_CH": [3,3],
            "WINDOW": 200,
            "CODE": 20,
            "CLASS":14
        }
    elif ds_name == "SLEEPEDF":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["EEG","EOG"],
            "MODALITY_CH": [1,1],
            "WINDOW": 3000,
            "CODE": 20,
            "CLASS":5,
            "HWIN": 2500,
            "FWIN": 500,
            "LABEL_NAME":["Awake","REM","N1","N2-N3","N4"]
        }

    elif ds_name == "SLEEPEDF4":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["EEG1","EEG2","EOG","EMG"],
            "MODALITY_CH": [1,1,1,1],
            "WINDOW": 3000,
            "CODE": 20,
            "CLASS":5,
            "HWIN": 2500,
            "FWIN": 500,
            "LABEL_NAME":["Awake","REM","N1","N2-N3","N4"]
        }
    elif ds_name == "PAMAP2":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["ACC1","ACC2","ACC3"],
            "MODALITY_CH": [3,3,3],
            "WINDOW": 512,
            "CODE": 20,
            "CLASS":9,
            "HWIN": 384,
            "FWIN": 128,
            "LABEL_NAME":["sitting", "standing", "walking", "running", "cycling", "nordic_walking", "ascending_stairs", "descending_stairs", "rope_jumping"]
        }
    elif ds_name == "PAMAP2_2":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["ACC1","ACC2"],
            "MODALITY_CH": [3,3],
            "WINDOW": 512,
            "CODE": 20,
            "CLASS":9,
            "HWIN": 384,
            "FWIN": 128,
            "LABEL_NAME":["sitting", "standing", "walking", "running", "cycling", "nordic_walking", "ascending_stairs", "descending_stairs", "rope_jumping"]
        }

    elif ds_name == "WESAD":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["ECG","EMG"],
            "MODALITY_CH": [1,1],
            "WINDOW": 1000,
            "CODE": 20,
            "CLASS":4,
            "HWIN": 800,
            "FWIN": 200,
            "LABEL_NAME":["Baseline","Stress","Amusement","Meditation"]
        }

    elif ds_name == "WESAD4":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["ECG","EMG","EDA"],
            "MODALITY_CH": [1,1,1],
            "WINDOW": 1000,
            "CODE": 20,
            "CLASS":4,
            "HWIN": 800,
            "FWIN": 200,
            "LABEL_NAME":["Baseline","Stress","Amusement","Meditation"]
        }
    elif ds_name == "SYN6":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["EEG1","EEG2","EEG3","EEG4","EEG5","EEG6"],
            "MODALITY_CH": [1,1,1,1,1,1],
            "WINDOW": 60,
            "CODE": 20,
            "CLASS":2,
            "HWIN" :48,
            "FWIN" :12,
            "LABEL_NAME":["Sitting", "Standing"]
        }
    elif ds_name == "SYN12":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["EEG1","EEG2","EEG3","EEG4","EEG5","EEG6","EEG7","EEG8","EEG9","EEG10","EEG11","EEG12"],
            "MODALITY_CH": [1,1,1,1,1,1,1,1,1,1,1,1],
            "WINDOW": 60,
            "CODE": 20,
            "CLASS":2,
            "HWIN" :48,
            "FWIN" :12,
            "LABEL_NAME":["Sitting", "Standing"]
        }

    elif ds_name == "SYN24":
        config_dict = {
            "DS_NAME": ds_name,
            "MODALITY_NAME": ["EEG1","EEG2","EEG3","EEG4","EEG5","EEG6","EEG7","EEG8","EEG9","EEG10","EEG11","EEG12","EEG13","EEG14","EEG15","EEG16","EEG17","EEG18","EEG19","EEG20","EEG21","EEG22","EEG23","EEG24"],
            "MODALITY_CH": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            "WINDOW": 60,
            "CODE": 20,
            "CLASS":2,
            "HWIN" :48,
            "FWIN" :12,
            "LABEL_NAME":["Sitting", "Standing"]
        }
    return config_dict
