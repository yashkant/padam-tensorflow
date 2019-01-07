import matplotlib.pyplot as plt
import pandas as pd

optim_params = {
    '0.1': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'b1': 0.9,
        'b2': 0.999,
        'name':'learning rate = 0.1',
        'color': 'green',
        'linestyle':'-'
    },
    '0.01': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'name':'learning rate = 0.01',
        'color': 'darkred',
        'linestyle':'--'
    },
    '0.001': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'name':'learning rate = 0.001',
        'color': 'blue',
        'linestyle':'-'
    }
}

p = '0.25'
param = ['loss','val_acc','val_top_k_categorical_accuracy']
lr_array = ['0.1','0.01', '0.001'] #loss;val_acc;val_top_k_categorical_accuracy
dataset = 'cifar100'
files = []

#label = {'loss':'Train Loss', 'val_acc':'Test Error', 'val_top_k_categorical_accuracy':'Test Error(top 5)'}
files={}
for l in lr_array:
    files[l]='log_p'+p+'_lr'+l+'_'+dataset+'.csv'

data = pd.DataFrame()

for l in lr_array:
    df = pd.read_csv(files[l], delimiter = ';')
    data[l] = df['val_acc']

#if param == 'val_acc' or param == 'val_top_k_categorical_accuracy':
data = 1-data
#print(data)
plt.figure()

for l in lr_array:
    op = optim_params[l]
    data[l].plot(label = op['name'], color=op['color'], linestyle=op['linestyle'])
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
plt.savefig('figure_p0.25_'+dataset+'_Test_Error_'+'.pdf')