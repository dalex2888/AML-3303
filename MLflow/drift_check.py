import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

train = pd.read_csv("data/train.csv")
production = pd.read_csv("data/test.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train, current_data=production)
report.save_html("drift_report.html")
