from angle_emb import AnglE, Prompts

angle = AnglE.from_pretrained('C:/Users/Administrator/.cache/torch/sentence_transformers/WhereIsAI_UAE-Large-V1', pooling_strategy='cls').cuda()
vecs = angle.encode(['hello world1', 'hello world2'], to_numpy=True)
print(vecs)



angle.set_prompt(prompt=Prompts.C)
vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
print(vecs)
