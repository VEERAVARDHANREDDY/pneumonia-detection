import graphviz

def create_architecture_diagram():
    dot = graphviz.Digraph(comment='Model Comparison Architecture', format='png')
    dot.attr(rankdir='LR', size='10,10', dpi='300')
    
    # Input Node
    dot.node('Input', 'Chest X-Ray Image', shape='box', style='filled', fillcolor='lightblue')
    
    # Backbone Models
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Deep Learning Backbones (Feature Extractors)', style='dashed')
        c.node('ResNet50', 'ResNet50', shape='component', style='filled', fillcolor='lightgrey')
        c.node('VGG16', 'VGG16', shape='component', style='filled', fillcolor='lightgrey')
        c.node('VGG19', 'VGG19', shape='component', style='filled', fillcolor='lightgrey')
        c.node('DenseNet121', 'DenseNet121', shape='component', style='filled', fillcolor='lightgrey')
        c.node('DenseNet169', 'DenseNet169', shape='component', style='filled', fillcolor='lightgrey')
        
    # Feature Extraction
    dot.node('Features', 'Feature Vectors\n(Global Avg Pooling)', shape='cylinder', style='filled', fillcolor='orange')
    
    # Classifiers
    with dot.subgraph(name='cluster_classifiers') as c:
        c.attr(label='Classifiers', style='dashed')
        c.node('SVM', 'SVM (RBF)', shape='ellipse', style='filled', fillcolor='lightgreen')
        c.node('RF', 'Random Forest', shape='ellipse', style='filled', fillcolor='lightgreen')
        c.node('kNN', 'k-NN', shape='ellipse', style='filled', fillcolor='lightgreen')
        c.node('NB', 'Naive Bayes', shape='ellipse', style='filled', fillcolor='lightgreen')
        
    # Output
    dot.node('Output', 'Diagnosis\n(Pneumonia / Normal)', shape='doublecircle', style='filled', fillcolor='gold')
    
    # Edges
    for model in ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169']:
        dot.edge('Input', model)
        dot.edge(model, 'Features')
        
    for clf in ['SVM', 'RF', 'kNN', 'NB']:
        dot.edge('Features', clf)
        dot.edge(clf, 'Output')
        
    # Render
    output_path = 'model_comparison_pipeline'
    dot.render(output_path, view=False)
    print(f"Diagram generated: {output_path}.png")

if __name__ == '__main__':
    try:
        create_architecture_diagram()
    except Exception as e:
        print(f"Error generating diagram: {e}")
        print("Please ensure Graphviz is installed on your system and in your PATH.")
