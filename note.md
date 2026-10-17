通过context_id可以将问题、证据、答案联系在一起。但在送给模型前，harness.py:201-244 的 _resolved_question() 会重新构造一个只包含允许字段的问题对象，主动排除答案
faiss 索引维度都是1536