#!/bin/bash
TAG="exp_ID-visualPT"     #exp_ID

LR="$1" 
IFS=',' read -ra LR <<< "$LR"
echo "LR: ${LR[@]}"

# Function to run the job
run_job() {
    OPTIM_SEED=${optim_seed} \
    VIS_ENCODER=${vis_encoder} \
    DATASET_NAME=${dataset_name} \
    SPLIT_SEED=${split_seed} \
    MODEL=${SETTING} \
    DATASET_DIR=${dataset_dir} \
    OUTPUT_DIR=${DIR} \
    LOSS_TYPE=${loss_type} \
    EPOCHS=${epoch_num} \
    LR=${lr} \
    DECAY=${decay} \
    BATCH_SIZE=${batch_size} \
    TEMPERATURE=${TEMPERATURE} \
    DATASET_DESCRIPTIONS=${dataset_descriptions} \
    image_alpha=${alpha} \
    image_temperature=${image_temperature} \
    W_gamma=${gamma} \
    W_k=${k} \
    tr_head=${tr_head} \
    tr_tail=${tr_tail} \
    accelerate launch --config_file methods_config/accelerate_localtest_config.yml run_main_${learning_paradigm}.py \
                      --model_config ${SETTING}_config_PPSB.yml --learning_paradigm ${learning_paradigm}
}


EPOCHS=(150)
DECAY=(0.05)
BATCH_SIZE=(32)
dataset_dirs=('data_')    #  add the path here containing datasets
vis_encoders=('ViT-B/32') #  'ViT-B/32' or 'ViT-L/14'
split_seeds=(500)         # This indicate the split for TRZSL, i.e., 500, 0, or 200. For other learning setting this is 500 as default.
dataset_names=('Flowers102' 'CUB' 'DTD' 'OxfordPets' 'FGVCAircraft' 'StanfordCars') #  'DTD' 'CUB' 'Flowers102' 'OxfordPets' 'FGVCAircraft' 'StanfordCars'
SETTINGS=('visual_fpl')   # Different prompt SETTINGS textual_fpl
optim_seeds=(1)         # 1 2 3 are the seeds we used
loss_types=('ce')           # Choose among different loss func: 'cc' (default) 'rc_rc' (RC) 'lw_lw' (LW) 'rc_cav' (CAV)

TEMPERATUREs=(1.0)
LAMBDAs=(1.0)

learning_paradigms=('ul' 'ssl' 'trzsl') # Choose among: ul, ssl, trzsl

# added parameters
DATASET_DESCRIPTIONS=(1)
image_alpha=(1.0)
image_temperature=(0.2)
W_gamma=(1.0)
W_k=(2)
Tr_head=(1.1)
Tr_tail=(0.9)

for dataset_dir in "${dataset_dirs[@]}"; do
for dataset_name in "${dataset_names[@]}"; do

if [ "$dataset_name" == "CUB" ]; then
    EPOCHS=(150) 
    DATASET_DESCRIPTIONS=(1)
    image_alpha=(1.0)
    image_temperature=(0.2)
    W_gamma=(1.0)
    W_k=(2)
elif [ "$dataset_name" == "DTD" ]; then
    EPOCHS=(150) 
    DATASET_DESCRIPTIONS=(1)
    image_alpha=(0.5)
    image_temperature=(0.2)
    W_gamma=(1.0)
    W_k=(2)
elif [ "$dataset_name" == "Flowers102" ]; then
    EPOCHS=(150) 
    DATASET_DESCRIPTIONS=(1)
    image_alpha=(1.0)
    image_temperature=(0.2)
    W_gamma=(1.0)
    W_k=(2)
elif [ "$dataset_name" == "FGVCAircraft" ]; then
    EPOCHS=(50) 
    DATASET_DESCRIPTIONS=(1)
    image_alpha=(1.0)
    image_temperature=(0.2)
    W_gamma=(1.0)
    W_k=(2)
elif [ "$dataset_name" == "OxfordPets" ]; then
    EPOCHS=(150) 
    DATASET_DESCRIPTIONS=(1)
    image_alpha=(1.0)
    image_temperature=(0.2)
    W_gamma=(1.0)
    W_k=(2)
elif [ "$dataset_name" == "StanfordCars" ]; then
    EPOCHS=(100) 
    DATASET_DESCRIPTIONS=(0)
    image_alpha=(1.0)
    image_temperature=(0.2)
    W_gamma=(1.0)
    W_k=(2)
else
    echo "Invalid Dataset name"
    exit 1
fi

for learning_paradigm in "${learning_paradigms[@]}"; do

for vis_encoder in "${vis_encoders[@]}"; do
for optim_seed in "${optim_seeds[@]}"; do
for split_seed in "${split_seeds[@]}"; do

for epoch_num in "${EPOCHS[@]}"; do
for lr in "${LR[@]}"; do
for decay in "${DECAY[@]}"; do
for batch_size in "${BATCH_SIZE[@]}"; do
for SETTING in "${SETTINGS[@]}"; do
for loss_type in "${loss_types[@]}"; do

for dataset_descriptions in "${DATASET_DESCRIPTIONS[@]}"; do
for alpha in "${image_alpha[@]}"; do
for image_temperature in "${image_temperature[@]}"; do
for gamma in "${W_gamma[@]}"; do
for k in "${W_k[@]}"; do
for tr_head in "${Tr_head[@]}"; do
for tr_tail in "${Tr_tail[@]}"; do

for TEMPERATURE in "${TEMPERATUREs[@]}"; do

    LOG_FILE="script_results/log_${TAG}_${dataset_name}.txt"
    total_iterations=$((${#EPOCHS[@]} * \
                        ${#LR[@]} * \
                        ${#DECAY[@]} * \
                        ${#BATCH_SIZE[@]} * \
                        ${#dataset_dirs[@]} * \
                        ${#vis_encoders[@]} * \
                        ${#split_seeds[@]} * \
                        ${#dataset_names[@]} * \
                        ${#SETTINGS[@]} * \
                        ${#optim_seeds[@]} * \
                        ${#loss_types[@]} * \
                        ${#TEMPERATUREs[@]} * \
                        ${#learning_paradigms[@]} * \
                        ${#image_alpha[@]} * \
                        ${#W_k[@]} * \
                        ${#Tr_head[@]} * \
                        ${#Tr_tail[@]} * \
                        ${#W_gamma[@]} * \
                        ${#DATASET_DESCRIPTIONS[@]}))

    echo "The loop will iterate $total_iterations times."

    common_id="dataset-${dataset_name}_tr_tail-${tr_tail}_tr_head-${tr_head}_k-${k}_gamma-${gamma}_alpha-${alpha}_descriptions-${dataset_descriptions}_setting-${SETTING}_lpardigm-${learning_paradigm}_encoder-${vis_encoder}_split-${split_seed}_seed-${optim_seed}_epoch-${epoch_num}_lr-${lr}_decay-${decay}_bs-${batch_size}_loss-${loss_type}_T-${TEMPERATURE}"
    DIR=./output/${dataset_name}/${SETTING}/${vis_encoder}_SplitSeed${split_seed}-${TAG}/SEED${optim_seed}/${common_id}
    
    if [ -d "$DIR" ]; then
        echo -e "------------\n Results are available in ${DIR}. Skip this job"
    else
        echo "======>>> Run this job and save the output to ${DIR}"
        run_job
        ACCURACY=$(grep 'Testset accuracy:' ${DIR}/log.txt | \
          awk -F': ' '{print $2}' | \
          sed 's/%//g' | \
          sort -n | \
          tail -1)
        RECORD="id: ${common_id} ----> test * accuracy: ${ACCURACY}"
        echo "${RECORD}" | tee -a ${LOG_FILE}
        echo "${RECORD}" >> ${DIR}/log.txt
    fi

done
done
done
done
done

done
done
done
done
done
done

done
done
done
done
done
done
done
done 
done 