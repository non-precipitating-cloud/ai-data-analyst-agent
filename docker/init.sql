-- 初始化：启用 pgvector 扩展。
-- 说明：knowledge_chunks 表由应用按当前 embedding 维度动态创建，
-- 以免与真实 OpenAI 向量（1536 维）或本地回退向量（384 维）产生维度冲突。
CREATE EXTENSION IF NOT EXISTS vector;
