import pandas as pd
import matplotlib.pyplot as plt

path = './model_bert-base-chinese_T01/training_progress_scores.csv'
df = pd.read_csv(path)
df['epoch'] = df.index + 1
print(df)

plot = df.plot.line(x='epoch', y=['train_loss', 'eval_loss'])
plt.title('NER')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'eval_loss'], loc='upper right')
plt.ylim(0.0, 1.0)
plt.show()
