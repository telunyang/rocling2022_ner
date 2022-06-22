import pandas as pd
import matplotlib.pyplot as plt

path = './outputs/bs-64-epo-10/training_progress_scores.csv'
df = pd.read_csv(path)
print(df)

plot = df.plot.line(x='global_step', y=['train_loss', 'eval_loss', 'f1_score'])
plt.title('NER')
plt.ylabel('loss')
plt.xlabel('step')
plt.legend(['training', 'validation', 'f1 score'], loc='lower right')
plt.show()
