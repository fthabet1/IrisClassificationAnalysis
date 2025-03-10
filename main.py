import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

def runModel(model, xTrain, xTest, yTrain, yTest):
    """Train and evaluate a model, return metrics"""
    # Start timer
    startTime = time()
    
    # Fit the model
    model.fit(xTrain, yTrain)
    
    # Make predictions
    predictions = model.predict(xTest)
    
    # Calculate metrics
    accuracy = accuracy_score(yTest, predictions)
    precision = precision_score(yTest, predictions, average='weighted')
    recall = recall_score(yTest, predictions, average='weighted')
    f1 = f1_score(yTest, predictions, average='weighted')
    
    # End timer
    trainingTime = time() - startTime
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1Score': f1,
        'trainingTime': trainingTime
    }


# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('IRIS.csv')
    
    # Split features and target
    X = df.iloc[:, :-1]
    Y = df['species']
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Number of folds for cross-validation
    kFolds = 5
    # Fixed random seed for reproducibility
    randomSeed = 42
    
    # Setup K-Fold cross-validation
    kf = KFold(n_splits=kFolds, shuffle=True, random_state=randomSeed)
    
    # Store results
    allResults = {modelName: {'accuracy': [], 'precision': [], 'recall': [], 
                              'f1Score': [], 'trainingTime': []} 
                 for modelName in models.keys()}
    
    print(f"Running each model with {kFolds}-fold cross-validation...")
    
    # Run each model with k-fold cross-validation
    for modelName, model in models.items():
        print(f"Evaluating {modelName}...")
        
        # Run k-fold cross-validation
        for foldNum, (trainIdx, testIdx) in enumerate(kf.split(X)):
            print(f"  Fold {foldNum+1}/{kFolds}")
            
            # Split data for this fold
            xTrain, xTest = X.iloc[trainIdx], X.iloc[testIdx]
            yTrain, yTest = Y.iloc[trainIdx], Y.iloc[testIdx]
            
            # Scale features if needed
            if modelName in ['Logistic Regression', 'SVM', 'KNN']:
                scaler = StandardScaler()
                xTrainScaled = scaler.fit_transform(xTrain)
                xTestScaled = scaler.transform(xTest)
                xTrainData, xTestData = xTrainScaled, xTestScaled
            else:
                xTrainData, xTestData = xTrain, xTest
                
            # Run model and collect results
            foldResults = runModel(model, xTrainData, xTestData, yTrain, yTest)
            
            # Store all metrics
            for metric, value in foldResults.items():
                allResults[modelName][metric].append(value)
    
    print("All cross-validation runs completed!")
    
    # Calculate average metrics
    avgResults = {}
    for modelName, metrics in allResults.items():
        avgResults[modelName] = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
        
        # Also calculate standard deviations
        avgResults[modelName].update({
            f"{metric}Std": np.std(values) for metric, values in metrics.items()
        })
    
    # Print average results
    print(f"\nAverage Results Over {kFolds}-Fold Cross-Validation:")
    print("-" * 50)
    
    for modelName, metrics in avgResults.items():
        print(f"\n{modelName}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ± {metrics['accuracyStd']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} ± {metrics['precisionStd']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f} ± {metrics['recallStd']:.4f}")
        print(f"  F1 Score: {metrics['f1Score']:.4f} ± {metrics['f1ScoreStd']:.4f}")
        print(f"  Average Training Time: {metrics['trainingTime']:.6f} seconds")
    
    # Find the best model based on average accuracy
    bestModel = max(avgResults.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest performing model: {bestModel} with average accuracy {avgResults[bestModel]['accuracy']:.4f}")
    
    # Visualize average metrics with error bars
    metricsToPlot = ['accuracy', 'precision', 'recall', 'f1Score']
    
    fig, axes = plt.subplots(len(metricsToPlot), 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Model Performance Comparison ({kFolds}-Fold Cross-Validation)', fontsize=16)
    
    for i, metric in enumerate(metricsToPlot):
        ax = axes[i]
        
        # Extract values and standard deviations
        modelsList = list(avgResults.keys())
        values = [avgResults[model][metric] for model in modelsList]
        errors = [avgResults[model][f"{metric}Std"] for model in modelsList]
        
        # Create bar plot with error bars
        bars = ax.bar(modelsList, values, yerr=errors, capsize=5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for the title
    plt.savefig('model_comparison_kfold.png')
    plt.show()
    
    # Also create a single comparison chart for the paper
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    barWidth = 0.2
    index = np.arange(len(models))
    
    # Get list of model names
    modelNames = list(models.keys())
    
    # Create bars for each metric
    plt.bar(index, [avgResults[model]['accuracy'] for model in modelNames], 
            barWidth, label='Accuracy', yerr=[avgResults[model]['accuracyStd'] for model in modelNames], capsize=3)
    
    plt.bar(index + barWidth, [avgResults[model]['precision'] for model in modelNames], 
            barWidth, label='Precision', yerr=[avgResults[model]['precisionStd'] for model in modelNames], capsize=3)
    
    plt.bar(index + 2*barWidth, [avgResults[model]['recall'] for model in modelNames], 
            barWidth, label='Recall', yerr=[avgResults[model]['recallStd'] for model in modelNames], capsize=3)
    
    plt.bar(index + 3*barWidth, [avgResults[model]['f1Score'] for model in modelNames], 
            barWidth, label='F1 Score', yerr=[avgResults[model]['f1ScoreStd'] for model in modelNames], capsize=3)
    
    # Add labels, title and legend
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title(f'Model Performance Comparison ({kFolds}-Fold Cross-Validation)')
    plt.xticks(index + 1.5*barWidth, modelNames, rotation=45)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison_combined_kfold.png')
    plt.show()
    
    # Create a table comparing training times
    plt.figure(figsize=(10, 4))
    plt.title('Average Training Time Comparison')
    
    # Extract model names and training times
    modelNamesList = list(models.keys())
    trainingTimes = [avgResults[model]['trainingTime'] for model in modelNamesList]
    
    # Create bar chart with model names as labels
    bars = plt.bar(modelNamesList, trainingTimes)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.6f}s', ha='center', va='bottom', fontsize=8)
    
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('training_time_comparison_kfold.png')
    plt.show()