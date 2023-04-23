#!/bin/bash

# Set the path to your Python executable if necessary
PYTHON=python

# Set the path to your train.py script
#=train_multilabel_final.py


# Set the path to your JSON configuration file
# CONFIG1=config1.json
# CONFIG2=config2.json
# CONFIG3=config3.json
# CONFIG4=config4.json
# CONFIG5=config5.json
# CONFIG6=config6.json
# CONFIG7=config7.json

# # Run the Python script with the JSON file as an argument
# python train_multilabel_final.py --config $CONFIG
# python train_multilabel_final.py --config $CONFIG2
# python train_multilabel_final.py --config $CONFIG3
# python train_multilabel_final.py --config $CONFIG4
# python train_multilabel_final.py --config $CONFIG5
# python train_multilabel_final.py --config $CONFIG6
# python train_multilabel_final.py --config $CONFIG7


#!/bin/bash

# Set the path to your train.py script
CONFIG = 

# Set the path to your JSON configuration file

# Run the Python script with the JSON file as an argument
$PYTHON train_multilabel_final.py --config $CONFIG
