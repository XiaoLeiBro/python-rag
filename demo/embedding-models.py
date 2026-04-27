# 嵌入模型demo
# 使用 HuggingFaceEmbeddings 对接 HuggingFace 模型
from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")

# 单条转换嵌入
vec = model.embed_query("你好")
print(f"查询向量维度: {len(vec)}, 前10维: {vec[:10]}")

# 批量转行嵌入
docs = ["好啊", "晚上吃什么", "今天天气不错", "你好吗"]
vecs = model.embed_documents(docs)


# 用余弦相似度匹配：查询向量 vs 文档向量
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)

scores = [(doc, cosine(vec, v)) for doc, v in zip(docs, vecs)]
scores.sort(key=lambda x: x[1], reverse=True)

print(f'查询: "你好"')
print("匹配结果（余弦相似度降序）:")
for rank, (doc, score) in enumerate(scores, 1):
    bar = "█" * int(score * 50)
    print(f"  {rank}. \"{doc}\"  {score:.4f}  {bar}")


