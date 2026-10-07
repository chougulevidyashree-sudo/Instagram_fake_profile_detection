import joblib
model=joblib.dump('model.pkl')
def digit_ratio(username):
    digits = sum(c.isdigit() for c in username)
    return digits / len(username)
def predict_profile(features):
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]

    confidence = max(probability) * 100   # convert to %

    return prediction, round(confidence, 2)

# def predict_profile(username, bio, followers, following, posts):
    
#     score = 0
    
#     # Rule-based simple logic (no heavy ML needed)
    
#     if digit_ratio(username) > 0.5:
#         score += 1
        
#     if len(bio) < 10:
#         score += 1
        
#     if followers < 50 and following > 300:
#         score += 1
        
#     if posts < 3:
#         score += 1

#     if score >= 2:
#         return "Fake Profile ❌"
#     else:
#         return "Real Profile ✅"