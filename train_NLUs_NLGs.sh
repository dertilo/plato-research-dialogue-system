
set -o errexit
rm -rf Models/CamRestNLG/Sys/* Models/CamRestNLG/Usr/* Models/CamRestNLU/Sys/* Models/CamRestNLU/Usr/*

ludwig experiment --model_definition_file Examples/config/ludwig_nlg_train.yaml --data_csv Data/data/DSTC2_NLG_sys.csv --output_directory Models/CamRestNLG/Sys/
ludwig experiment --model_definition_file Examples/config/ludwig_nlg_train.yaml --data_csv Data/data/DSTC2_NLG_usr.csv --output_directory Models/CamRestNLG/Usr/
ludwig experiment --model_definition_file Examples/config/ludwig_nlu_train.yaml --data_csv Data/data/DSTC2_NLU_sys.csv --output_directory Models/CamRestNLU/Sys/
ludwig experiment --model_definition_file Examples/config/ludwig_nlu_train.yaml --data_csv Data/data/DSTC2_NLU_usr.csv --output_directory Models/CamRestNLU/Usr/
