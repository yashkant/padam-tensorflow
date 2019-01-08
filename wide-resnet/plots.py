import matplotlib.pyplot as plt
import pandas as pd

optim_params = {
    'padam': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'b1': 0.9,
        'b2': 0.999, 
        'color': 'darkred',
        'linestyle':'-'
    },
    'adam': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'orange',
        'linestyle':'--'
    },
    'adamw': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'magenta',
        'linestyle':'--'
    },
    'amsgrad': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color' : 'darkgreen',
        'linestyle':'-.'
    },
    'sgd': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'm': 0.9,
        'color': 'blue',
        'linestyle':'-'
    }
}

parameter = ['loss','val_acc','val_top_k_categorical_accuracy'] #loss;val_acc;val_top_k_categorical_accuracy
optimizers = ['adam', 'sgd', 'amsgrad', 'padam']
dataset = 'cifar10'
files = []

label = {'loss':'Train Loss', 'val_acc':'Test Error', 'val_top_k_categorical_accuracy':'Test Error(top 5)'}

for optim in optimizers:
    files.append('log_' + optim + '_'+ dataset + '.csv')

for param in parameter:        
    
    data = pd.DataFrame()
    
    for f in range(len(files)):
        df = pd.read_csv(files[f], delimiter = ';')
        data[optimizers[f]] = df[param]
    
    if param == 'val_acc' or param == 'val_top_k_categorical_accuracy':
        data = 1-data
        
    plt.figure()
    for optimizer in optimizers:
        op = optim_params[optimizer]
        data[optimizer].plot(color=op['color'], linestyle=op['linestyle'])
    if param=='loss':
        y_lim = 1.6
    elif param == 'val_acc':
        y_lim = 0.9
    else:
        y_lim= 0.35
        
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel(label[param])
    plt.ylim(top=y_lim)
    #plt.show()
    plt.savefig('figure_'+dataset+'_'+label[param]+'.pdf')