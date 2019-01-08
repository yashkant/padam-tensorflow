import matplotlib.pyplot as plt
import pandas as pd

optim_params = {
    '0.0625': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'b1': 0.9,
        'b2': 0.999,
        'name':'p = 1/16',
        'color': 'green',
        'linestyle':'-'
    },
    '0.125': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'name':'p = 1/8',
        'color': 'orange',
        'linestyle':'-'
    },
    '0.25': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'name':'p = 1/4',
        'color': 'blue',
        'linestyle':'-'
    }
}


p_array = ['0.0625','0.125', '0.25'] #loss;val_acc;val_top_k_categorical_accuracy
dataset = 'cifar100'
files = []

#label = {'loss':'Train Loss', 'val_acc':'Test Error', 'val_top_k_categorical_accuracy':'Test Error(top 5)'}
files={}
for p in p_array:
    files[p]='log_'+p+'_'+dataset+'.csv'

data = pd.DataFrame()

for p in p_array:
    df = pd.read_csv(files[p], delimiter = ';')
    data[p] = df['val_acc']

#if param == 'val_acc' or param == 'val_top_k_categorical_accuracy':
data = 1-data
#print(data)
plt.figure()

for p in p_array:
    op = optim_params[p]
    data[p].plot(label = op['name'], color=op['color'], linestyle=op['linestyle'])
"""
if param=='loss':
    y_lim = 1.5
elif param == 'val_acc':
    y_lim = 0.7
else:
    y_lim= 0.20
"""
    
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Test Error')
#plt.ylim(top=y_lim)
#plt.show()
plt.savefig('figure_p_val_'+dataset+'.pdf')