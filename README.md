https://github.com/nguyenanhtuan1008/vgg16_svm_classification
Step1: Generate data to text file
(tensorflow-gpu) PS E:\acuity\tuan_experiment\svm\Gender-Classification> python .\generate_db.py gender_dataset_face db

Step2: extract_feature to train_x, test_x folder
changle line 29: save_path = img_path.replace("gender_dataset_face", "test_x")
to test_x or trainx:
(tensorflow-gpu) PS E:\acuity\tuan_experiment\svm\Gender-Classification> python .\extract_features.py db\train.txt
(tensorflow-gpu) PS E:\acuity\tuan_experiment\svm\Gender-Classification> python .\extract_features.py db\test.txt

Step3: Train model
(tensorflow-gpu) PS E:\acuity\tuan_experiment\svm\Gender-Classification> python .\train_model.py

Step4: Test images
(tensorflow-gpu) PS E:\acuity\tuan_experiment\svm\Gender-Classification> python .\GenderClassification_Image.py