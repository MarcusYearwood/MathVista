cd ../evaluation

##### multimodal-internlm #####
# generate solution
python generate_response.py \
--model internlm \
--output_dir ../results/internlm \
--output_file output_internlm.json

# extract answer
python extract_answer.py \
--output_dir ../results/internlm \
--output_file output_internlm.json 

# calculate score
python calculate_score.py \
--output_dir ../results/internlm \
--output_file output_internlm.json \
--score_file scores_internlm.json
