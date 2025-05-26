import os, cv2, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from lens_classifier import detect_ar_coating_interference, detect_edge_bevel_signature, check_circle_contour
from analysis.gabor_analysis import extract_gabor_features
from analysis.specular_flow import extract_specular_flow_features

X, y = [], []
for label, path in [("lens","data/train/lens"),("not_lens","data/train/not_lens")]:
    for fn in os.listdir(path):
        img = cv2.imread(os.path.join(path,fn))
        if img is None: continue
        gvec = extract_gabor_features(img)
        fvec = extract_specular_flow_features([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)])
        feats = [
            detect_ar_coating_interference(img),
            detect_edge_bevel_signature(img),
            check_circle_contour(img),
            float(np.std(gvec)),
            float(fvec[0])
        ]
        X.append(feats)
        y.append(1 if label=="lens" else 0)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X,y)
joblib.dump(clf, "lens_vs_not_lens.joblib")
print("Trained RF accuracy:", clf.score(X,y)) 