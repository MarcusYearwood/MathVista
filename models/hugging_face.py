import torch
from transformers import AutoModel, AutoTokenizer


# build gpt class
class HF_Model():
    def __init__(self, model="internlm/internlm-xcomposer2d5-7b", num_beams=3, patience=1000000, sleep_time=0):
        self.model = AutoModel.from_pretrained(model, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.model.eval()
        self.num_beams = num_beams
        self.patience = patience
        self.sleep_time = sleep_time

    def get_response(self, image_path, user_prompt):
        query = user_prompt
        response, his = self.model.chat(self.tokenizer, query, [image_path], do_sample=False, num_beams=3, use_meta=True)
        prediction = response.strip()
        return prediction
        
#         while patience > 0:
#             patience -= 1
#             try:
#                 # print("self.model", self.model)
#                 response, his = model.chat(self.tokenizer, query, image_path, do_sample=False, num_beams=3, use_meta=True)
#                 if self.n == 1:
#                     prediction = response.strip()
#                     if prediction != "" and prediction != None:
#                         return prediction
#                 else:
#                     prediction = [choice['message']['content'].strip() for choice in response['choices']]
#                     if prediction[0] != "" and prediction[0] != None:
#                         return prediction
#                     return prediction
                        
#             except Exception as e:
#                 print(e)
#                 # if "limit" not in str(e):
#                 #     print(e)
#                 # if "Please reduce the length of the messages." in str(e):
#                 #     print("!!Reduce user_prompt to", user_prompt[:-1])
#                 #     return ""
#                 if self.sleep_time > 0:
#                     time.sleep(self.sleep_time)
#         return ""
