problem_definition = {
    'task_type': 'binary_classification',
    'target_variable': 'churn',
    'primary_metric': 'recall',
    'secondary_metric': 'precision',  # to avoid many FP
    'minimum_threshold': 0.85,  # 85% recall
    'business_priority': 'Detect maximum employees at risk of leaving',
    'constraints': {
        'interpretability': 'High (HR needs to understand WHY)',
        'retraining_frequency': 'Quarterly or on-demand',
        'deployment': 'CSV reports → Dashboard (future)'
    },
    'cost_analysis': {
        'false_negative_cost': 'High ($15K-$50K replacement cost + project delays)',
        'false_positive_cost': 'Low (wasted retention resources, but employee stays)'
    }
}