import pickle
model = 'scibert_lstm'
model_dir = 'scibert_lstm'
# file = open(f'./rhetorical_roles/{model_dir}/results/f1_met_{model}_epoch5.pkl', 'rb')
file = open(f'/home/preethamthava/LegalEval2022/rhetorical_roles/{model_dir}/results/f1_met_{model}_epoch5.pkl', 'rb')
df = pickle.load(file)
file.close()
print(df)