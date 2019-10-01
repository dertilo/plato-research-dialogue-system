### setup
#### 1. run docker-container
    
    docker run --shm-size 8G --runtime=nvidia --rm -it -v /home/gunther/tilo_data:/docker-share plato:latest bash
#### 2. manually install ludwig dependency (needs password to TUB-gitlab)
    pip install git+https://<USER_NAME>@gitlab.tubit.tu-berlin.de/OKS/ludwig.git
#### 3. download dialog-systems-challenge-2-data
    python download_data.py 
#### 4. process data
TODO: not working anymore??

    python runDSTC2DataParser.py -data_path dstc2_data/data/  

### Ludwig NLG
    ludwig experiment --model_definition_file Examples/config/ludwig_nlg_train.yaml --data_csv Data/data/DSTC2_NLG_sys.csv --output_directory Models/CamRestNLG/Sys/
    ludwig experiment --model_definition_file Examples/config/ludwig_nlg_train.yaml --data_csv Data/data/DSTC2_NLG_usr.csv --output_directory Models/CamRestNLG/Usr/

### Ludwig NLU
    add line:     tf_config.gpu_options.allow_growth = True to ludwig/ludwig/utils/tf_utils.py
    ludwig experiment --model_definition_file Examples/config/ludwig_nlu_train.yaml --data_csv Data/data/DSTC2_NLU_sys.csv --output_directory Models/CamRestNLU/Sys/
    ludwig experiment --model_definition_file Examples/config/ludwig_nlu_train.yaml --data_csv Data/data/DSTC2_NLU_usr.csv --output_directory Models/CamRestNLU/Usr/

### plato dialog policy
    python runPlatoRDS.py -config Examples/config/CamRest_MA_train_pretrained_NLU_NLG.yaml
    
-> not really working?
    
### train seq2seq chitchat-model via ludwig on metalwoz-data

manually download [metalwoz](https://www.microsoft.com/en-us/research/project/metalwoz/) 
    
    python runMetalWOZDataParser.py -data_path ../metalwoz-v1/dialogues/ORDER_PIZZA.txt

    ludwig train -g 0,1 --data_csv Data/data/metalwoz.csv --model_definition_file Examples/config/metalWOZ_seq2seq_ludwig.yaml --output_directory "Models/JointModels/"

    python runPlatoRDS.py -config Examples/config/metalwoz_generic.yaml

## Paper: Collaborative Multi-Agent Dialogue Model Training Via Reinforcement
Learning [Papangelis 2019]
* DSTC2 as seed data, we trained (NLU) and (NLG) 
* Win or Lose Fast Policy Hill Climbing (WoLF-PHC)
* only observing the other agent’s language output and a reward signal.
