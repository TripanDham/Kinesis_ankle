import joblib
data = joblib.load('/media/tripan/Data/DDP/amputee_data/training_data_combined/processed_motions.joblib')
key = list(data.keys())[0]
bxpos = data[key]['body_xpos']
print("First frame body_xpos for key:", key)
print(bxpos[0])
