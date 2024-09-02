import numpy as np 
import matplotlib.pyplot as plt 



def plot_boxes(df):
    models = df.columns
    metrics = ['ACC', 'Recall', 'Precision', 'F1','MCC','AUC']
    for i in range(len(metrics)):
        
        metric_scores_of_models = [np.stack(df[model].values)[:, i] for model in models] 
        
        
        fig, ax = plt.subplots(figsize=(10, 7))  # Adjust the size as needed
    
        # Add an axes to the figure
        #ax = fig.add_axes([0, 0, 1, 1])  # Full figure
        
        # Create a box plot on the axes
        bp = ax.boxplot(metric_scores_of_models, positions=np.arange(1, len(models)+1))
        
        # Customize plot
        ax.set_xticks(np.arange(1, len(models)+1))  # Set the x-ticks to be at the positions of the boxes
        ax.set_xticklabels(models)  # Label the x-ticks
        ax.set_title(f"Box plots of {metrics[i]} across models.")
        ax.set_ylabel(metrics[i])
        fig.tight_layout()
        ax.set_xticklabels(models, rotation=45, ha='right')
        plt.savefig(f'results/figs/{metrics[i]}box_plot.png')
        # show plot
        plt.show()

        
def plot_rocs(rocs, name):
    mean_fpr = np.linspace(0, 1, 100)
    models = list(rocs.keys())
    fig, ax = plt.subplots(figsize=(10, 7))
    #ax = fig.add_axes([0, 0, 1, 1])
    for model in models:
        mean_tpr = np.mean(rocs[model], axis = 0)
        ax.plot(mean_fpr,mean_tpr, label= model, lw = 2, alpha = 0.8)
    ax.set_title(f"Mean ROC plots using dataset {name}.")
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'results/figs/{name}ROC_plot.png')
    plt.show()