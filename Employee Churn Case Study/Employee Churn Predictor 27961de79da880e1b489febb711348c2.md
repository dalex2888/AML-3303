# Employee Churn Predictor

*Predictive Analytics for Workforce Retention*

---

**Prepared by:**

Diego Alexander Espinosa Carreño

**Department:**

Research & Development - *Artificial Intelligence & Machine Learning*

**Submission Date:**

September 2025

# Project Overview

TechNova Solutions, a mid-sized IT services company with ~1,200 employees, has been facing an attrition rate well above industry standards. Despite offering competitive salaries and benefits, the company struggles to retain talent, particularly in technical and client-facing roles. This turnover has increased costs, delayed projects, and reduced overall employee satisfaction.  

The Employee Churn Predictor is intended to predict whether an employee is about to leave the company by implementing classification models trained on historical data. A secondary goal is to identify key factors that contribute to churn, enabling the company to design targeted strategies that improve employee retention.

In practice, the system will generate churn probability scores for individual employees and highlight the most influential factors driving attrition. These insights will be provided to HR and management teams, enabling proactive retention measures such as targeted engagement programs, career development opportunities, or workload adjustments. Over time, the solution can be scaled into an integrated HR analytics dashboard, supporting continuous monitoring of employee retention trends and helping the company reduce costs and improve workforce stability.

# Objectives & Scope

- Identify the most relevant factors that contribute to employee attrition.
- Build a predictive model that can classify whether an employee is likely to leave with a Recall above 85%.
- Provide actionable recommendations that can help HR improve retention and employee satisfaction.

Requirements not mentioned in this section are out of the scope, however, future stages are proposed below as future work for this especification.

## Data Sources

TechNova Solutions preserves employee-related data in a relational cloud-based database. Data is collected using SQL transactions via API to ensure reliability and security, then stored in an offline *Feature Store* for processing. Because the problem concerns workforce retention, the queries primarily affect HR modules and related systems. The information collected includes:

- **Personal information:** age, gender, dwelling, education, marital status.
- **Job information:** role, evaluations, salary, department, hours worked, work location, training hours, overtime, monthly hours average, projects, absenteeism, performance scores.

Most of the data gathered is numerical, with categorical attributes also represented. Due to the relatively low volume and variety, datasets are static and do not require continuous collection or processing. Feature descriptions are defined in the data dictionary, with additional features created during the engineering process.

To ensure data reliability, preprocessing applies standard quality checks. Missing values are imputed or removed depending on relevance, duplicates are eliminated, and outliers are assessed to prevent distortions. Engineered variables such as average monthly hours and tenure categories further enrich the model’s predictive capacity. For now, data is collected as periodic snapshots, but future iterations may adopt real-time pipelines once the initial model proves effective.

[employee_churn_data_dictionary.csv](data/employee_churn_data_dictionary.csv)

## Methodology & Pipeline Diagram

Deployment of the Churn Predictor follows the MLOps lifecycle to ensure scalability, reproducibility, and continuous improvement. The pipeline begins with an API connected to TechNova’s relational database, which feeds employee data into a Feature Store. Here, data undergoes preprocessing, validation, and feature engineering before model training.

**Initial Stage:**

- Model predictions are exported as CSV reports with churn probabilities and the most influential factors.
- HR can prioritize at-risk employees and test intervention strategies.

**Intermediate Stage:**

- **MLflow** is used for experiment tracking, logging model runs, parameters, and metrics for reproducibility.
- Visual dashboards may be introduced to provide HR with near real-time insights.

**Long-Term Stage:**

- Full integration into an HR analytics dashboard with automated pipelines.
- Scheduled or on-demand predictions, automated retraining, and performance monitoring.
- Role-based access and alerts ensure data security and continuous adaptation to workforce changes.

This approach moves the project from a simple CSV-based pilot to a production-ready MLOps ecosystem aligned with the project timeline and milestones.

![Simple Project Pipeline.drawio.png](/img/Simple_Project_Pipeline.drawio.png)

***Figure 1. ML Pipeline***

## Tools & Infrastructure

The technology stack for the project includes:

- **Programming Environment & Libraries**
    - *Python 3.13 or higher*: primary programming language for model development.
        - *Pandas, NumPy*: data manipulation, cleaning, and numerical computations.
        - *Scikit-learn*: classification models, feature engineering, and evaluation metrics.
        - *Matplotlib, Seaborn*: EDA and visualization.
- **Data Access & Storage**
    - *SQL*: data extraction from HR systems through secure queries.
    - *APIs*: for reliable, automated data retrieval.
    - *CSV*: intermediate storage format for preprocessing and modelling.
    - *Role-based access control and encryption*: ensures compliance with data privacy standards for sensitive HR information.
- **Environment & Workflow**
    - *Jupyter Notebook, VS Code*: interactive exploration, prototyping, and code development.
    - *GitHub*: version control, collaborative code management, and documentation.
    - *Jira*: project tracking and sprint planning.
    - *Notion*: team documentation and knowledge base.
- **Model Lifecycle & Deployment**
    - *Local training*: initial development and prototyping.
    - *Experiment tracking*: manual logging during prototyping; MLflow can be integrated in later stages.
    - *Output delivery*: initial predictions as CSV reports for HR; long-term integration into an HR analytics dashboard for continuous monitoring.

### Model Design and Evaluation

The predictive model will use **Random Forest** as the primary algorithm, with hyperparameters optimized via **GridSearch** to handle class imbalance and prevent overfitting. A secondary approach could explore **XGBoost** or a hybrid ensemble for comparison. Features are selected based on relevance from the Feature Store, with additional engineered variables (e.g., tenure categories, average monthly hours) to improve predictive power.

Model performance will be evaluated using **Recall** to prioritize high-risk employees, **F1 score** to balance precision and recall, and **learning curves** to detect underfitting or overfitting. Experiment tracking via **MLflow** will ensure reproducibility and facilitate comparison of different model configurations.

### Actionable Insights

The model outputs churn probabilities for individual employees and identifies the top factors influencing attrition. These insights will be delivered initially as **CSV reports**, allowing HR to prioritize at-risk employees and test targeted retention strategies. Over time, the system can feed a **dashboard** for continuous monitoring, supporting interventions such as engagement programs, workload adjustments, or career development initiatives.

## Risks & Mitigation

**Class Imbalance**

Although the overall attrition rate is high, the number of resignations may be imbalanced relative to the total employee population, which can affect the model’s ability to generalize and correctly identify at-risk employees. To mitigate this:

- **Data-Level:** Balance the dataset with oversampling, undersampling, or techniques like SMOTE to generate synthetic examples for underrepresented cases.
- **Model-Level:** Use ensemble algorithms such as Random Forest, with hyperparameters tuned to handle imbalance effectively.
- **Evaluation Metrics:** Focus on **Recall** to capture high-risk employees, while using **F1 score** to balance precision and recall and limit false positives.

**Underfitting / Overfitting**

Careful preprocessing, feature engineering, and sampling will help avoid underfitting. Hyperparameters, especially for Random Forest, will be optimized using GridSearch. Learning curves and feature importance plots will be monitored to ensure the model generalizes well and captures meaningful patterns rather than noise.

## Timeline & Expected Outcomes

The timeline below depicts a simplistic route to develop the proposal with the corresponding roles:

| **Phase** | **Duration** | **Key Activities** | **Roles** |
| --- | --- | --- | --- |
| **Data Preparation** | 2 weeks | Data extraction, cleaning, preprocessing, feature engineering, store in Feature Store | Data Engineer, Project Lead, HR Analyst |
| **Model Development & Evaluation** | 3 weeks | Train Random Forest (and alternative), hyperparameter tuning, evaluation (Recall, F1), MLflow logging | Project Lead, ML Engineer |
| **Pilot Deployment** | 2 weeks | Export CSV reports, provide initial actionable insights to HR | Project Lead, ML Engineer, HR Analyst |
| **Dashboard & MLOps Integration** | 3 weeks | Build dashboards, automate pipelines, set up monitoring and retraining | ML Engineer, UI/UX Developer, Project Lead |
| **Final Review & Handover** | 1 week | Validate outputs, document methodology, deliver final report and dashboards | Project Lead, HR Analyst |

The following table translates the project’s objectives and scope into concrete, actionable outcomes. Each outcome aligns with a specific deliverable, providing clear value to both the technical implementation and HR decision-making processes.

| **Outcome** | **Description / Deliverable** | **Importance** |
| --- | --- | --- |
| **Predictive Model** | Individual churn risk scores (probabilities) for all employees. | Allows stakeholders to assess each employee’s risk of attrition and set thresholds for action. |
| **Feature Importance Analysis** | Identifies the top factors driving attrition. | Provides HR and managers with actionable insights into the key variables affecting turnover, supporting informed decision-making. |
| **Actionable HR Insights** | Highlights employees who may benefit from engagement programs, workload adjustments, or career development initiatives. | Enables targeted interventions to improve retention and employee satisfaction. |
| **Evaluation Metrics** | Model performance metrics, including Recall and F1 score. | Ensures reliable predictions, allowing HR to focus on high-risk employees without overlooking others. |