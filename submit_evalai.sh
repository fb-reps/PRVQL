# Define new output directory path
NEW_OUTPUT_DIR="./output"

# Define new cpt_path value
NEW_CPT_PATH="$NEW_OUTPUT_DIR/ego4d_egotracks/train/train/cpt_best_prob.pth.tar"

# Path to the Yaml configuration file
CONFIG_FILE="./config/eval.yaml"

# Use the sed command to update the output_dir value in the configuration file
sed -i "s|output_dir: .*|output_dir: $NEW_OUTPUT_DIR|" $CONFIG_FILE

# Use the sed command to update the cpt_path value in the configuration file
sed -i "s|cpt_path: .*|cpt_path: $NEW_CPT_PATH|" $CONFIG_FILE

# predict
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_predict.py --cfg  ./config/eval.yaml --eval

# get .json lastest track
python inference_results.py --cfg ./config/eval.yaml --eval
