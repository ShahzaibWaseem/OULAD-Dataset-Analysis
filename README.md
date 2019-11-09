# OULAD Dataset Analysis

## Dataset used: [Link]([https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip)

## Steps
1. Fetch data based on student id from specified files (Preprocessing of data). 
2. Based on fetched data (i.e. student activity) create labels using clustering (KNN) for each student that he will "procrastinate" (delay the homework yes or no). Add this value yes or no in csv file as a new column for each student.
3. Use this updated csv file and apply SVM, Logistic regression, ANN for training and testing to predict procrastination in students.
4. Take user input (values for all the columns used for training of model) from console and predict based on the provided input if particular student will procrastinate or not (this step is also testing of trained machine learning model).
5. Creation of graphs that shows the student records with labels of procrastination "True" or "False".
6. Accuracies of each trained and test model.

## Columns considered
- StudentInfo: id_student, highest_education, studied_credits, num_of_previous_attempts, final_result, disability
- StudentAssessment: date_submitted, score
- StudentVle: sum_of_sum_click (generated from summing over sum_clicks)