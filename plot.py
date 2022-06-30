import pandas as pd
import matplotlib.pyplot as plt

path = './outputs_bert/training_progress_scores.csv'
df = pd.read_csv(path)
print(df)

plot = df.plot.line(x='global_step', y=['train_loss', 'eval_loss'])
plt.title('NER')
plt.ylabel('loss')
plt.xlabel('step')
plt.legend(['training', 'validation'], loc='upper right')
plt.ylim(0.0, 1.0)
plt.show()
