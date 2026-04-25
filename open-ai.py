# import os
#
# from dotenv import load_dotenv
# from openai import OpenAI
#
# load_dotenv()
#
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
#
# completion = client.chat.completions.create(
#     model="qwen3.6-plus",
#     messages=[
#         {"role": "system", "content": "你是一个AI助手."},
#         {"role": "user", "content": "你是什么模型？"},
#     ],
#     stream=False,
# )
# # print(completion.choices[0].message.content)
