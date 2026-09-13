-- 初始化：启用 pgvector 扩展。
-- 说明：knowledge_chunks 表由应用按当前 embedding 维度动态创建，
-- 以免与真实 OpenAI 向量（1536 维）或本地回退向量（384 维）产生维度冲突。
CREATE EXTENSION IF NOT EXISTS vector;


# EXTENSION是给数据库安装的一个扩展，可以给数据库添加一批新的函数、类型、操作符
# ector是pgvector扩展提供的数据类型，用来在数据库里存向量（embedding）
# 向量的定义：一串固定长度的浮点数，比如[0.12, -0.35, 0.88, ...]
# 文本经过embedding模型会被转成这样的数字数组，语义详尽的文本，向量间距离越近
