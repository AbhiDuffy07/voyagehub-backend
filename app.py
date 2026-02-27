from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import json
import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"
# ── In-memory cache ───────────────────────────────────────────
recommendations_cache = {}

def get_cached(key):
    return recommendations_cache.get(key)

def set_cache(key, value):
    recommendations_cache[key] = value

# ── Load dataset once at startup ──────────────────────────────
with open(os.path.join(os.path.dirname(__file__),
    'destinations_with_real_attractions_final.json'), 'r', encoding='utf-8') as f:
    ALL_DESTINATIONS = json.load(f)

print(f"✅ Loaded {len(ALL_DESTINATIONS)} destinations")

def get_city_attractions(city_name):
    for dest in ALL_DESTINATIONS:
        if dest['city'].lower() == city_name.lower():
            return dest.get('attractions', [])
    return []

def call_gemini_with_retry(prompt, retries=3, wait=40):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if '429' in err:
                if attempt < retries - 1:
                    print(f"Rate limit hit. Waiting {wait}s... retry {attempt+1}/{retries}")
                    time.sleep(wait)
                else:
                    raise Exception("Rate limit exceeded. Please wait a minute.")
            else:
                raise e

def clean_json_response(raw):
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                return part
    return raw.strip()


# ════════════════════════════════════════════════════════════════
# ML SETUP — Runs once at startup
# ════════════════════════════════════════════════════════════════

# ── ML Algorithm 1: TF-IDF Vectorizer (Content-Based Filtering) ─
def build_tfidf_matrix(destinations):
    corpus = []
    for d in destinations:
        text = (
            f"{d.get('city','')} {d.get('country','')} "
            f"{d.get('continent','')} "
        )
        attrs = d.get('attractions', [])
        if isinstance(attrs, list):
            text += " ".join([
                a.get('name','') + " " + a.get('type','')
                for a in attrs[:5]
            ])
        corpus.append(text.strip())
    if not corpus:
        return None, None
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

TFIDF_VECTORIZER, TFIDF_MATRIX = build_tfidf_matrix(ALL_DESTINATIONS)
print("✅ TF-IDF matrix built")

# ── ML Algorithm 2: Popularity Score (computed on demand) ───────
# Formula: score = rating * log(1 + totalAttractions)
# Pre-compute at startup for speed
for d in ALL_DESTINATIONS:
    rating = float(d.get('rating', 3.0))
    attractions = len(d.get('attractions', [])) or int(d.get('totalAttractions', 1))
    d['_popularityScore'] = round(rating * np.log(1 + attractions), 4)

TRENDING_CACHE = sorted(
    ALL_DESTINATIONS,
    key=lambda x: x['_popularityScore'],
    reverse=True
)[:20]
print("✅ Popularity scores computed")

# ── ML Algorithm 3: Linear Regression Budget Predictor ──────────
EXPENSIVE_CITIES = {
    "paris", "london", "tokyo", "new york", "dubai",
    "sydney", "singapore", "zurich", "amsterdam",
    "venice", "rome", "barcelona", "copenhagen", "oslo"
}
EUROPE_COUNTRIES = {
    "france", "italy", "germany", "spain", "uk",
    "netherlands", "portugal", "greece", "switzerland",
    "austria", "belgium", "sweden", "norway", "denmark"
}
ASIA_COUNTRIES = {
    "japan", "india", "thailand", "vietnam", "china",
    "indonesia", "malaysia", "philippines", "singapore",
    "cambodia", "myanmar", "nepal", "sri lanka"
}

def train_budget_model():
    # Features: [days, members, is_expensive_city, is_europe, is_asia]
    training = [
        [3, 1, 1, 1, 0, 800],
        [5, 2, 1, 1, 0, 2500],
        [7, 4, 1, 1, 0, 5500],
        [3, 1, 0, 0, 1, 300],
        [5, 2, 0, 0, 1, 800],
        [7, 4, 0, 0, 1, 1800],
        [3, 1, 1, 0, 0, 600],
        [5, 2, 1, 0, 0, 1800],
        [7, 2, 0, 1, 0, 2200],
        [3, 2, 0, 0, 0, 400],
        [10, 2, 1, 1, 0, 4000],
        [4, 3, 0, 0, 1, 900],
        [6, 1, 1, 1, 0, 1500],
        [2, 2, 0, 0, 0, 250],
        [14, 2, 0, 0, 1, 2800],
        [7, 1, 1, 1, 0, 1800],
        [5, 4, 0, 0, 1, 1400],
        [3, 2, 1, 1, 0, 1200],
        [4, 1, 0, 0, 0, 350],
        [10, 3, 1, 1, 0, 5000],
        [6, 2, 0, 0, 0, 700],
        [8, 2, 1, 0, 0, 2400],
        [5, 1, 0, 0, 1, 500],
        [3, 4, 0, 0, 1, 900],
        [7, 2, 1, 1, 0, 2800],
    ]
    X = [[r[0], r[1], r[2], r[3], r[4]] for r in training]
    y = [r[5] for r in training]
    model = LinearRegression()
    model.fit(X, y)
    return model

BUDGET_MODEL = train_budget_model()
print("✅ Budget Linear Regression model trained")


# ════════════════════════════════════════════════════════════════
# EXISTING ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/generate-trip', methods=['POST'])
def get_travel_suggestions():
    try:
        data = request.json
        destination = data.get('destination', '')
        days = data.get('days', 3)
        interests = data.get('interests', [])
        num_people = data.get('numPeople', 1)
        budget = data.get('budget', 1000)

        cache_key = f"trip_{destination}_{days}_{budget}_{num_people}"
        cached = get_cached(cache_key)
        if cached:
            print(f"Cache hit for trip: {destination}")
            return jsonify({'success': True, 'itinerary': cached, 'from_cache': True})

        attractions = get_city_attractions(destination)

        if attractions:
            ppd = round(budget / max(num_people, 1) / max(days, 1), 1)
            if ppd < 30:
                max_attr = min(len(attractions), max(days * 2, 4))
            elif ppd < 80:
                max_attr = min(len(attractions), max(days * 3, 6))
            else:
                max_attr = min(len(attractions), days * 4)

            limited = attractions[:max_attr]
            attractions_text = "\n".join([
                f"- {a['name']} ({a['type']}) — Rating: {a['rating']} — {a['description']}"
                for a in limited
            ])
            budget_type = "budget-friendly" if ppd < 50 else "mid-range" if ppd < 150 else "luxury"

            prompt = f"""You are a travel planner. Create a detailed {days}-day itinerary for {destination}.

BUDGET CONTEXT:
- Total: ${budget} for {num_people} people = ${ppd}/person/day ({budget_type} travel)
- Distribute evenly: ~{max(1, max_attr // days)} attractions per day

RULES:
1. ONLY use attractions from the list below
2. Spread across ALL {days} days
3. Interests: {', '.join(interests) if interests else 'general tourism'}

AVAILABLE ATTRACTIONS IN {destination.upper()}:
{attractions_text}

Return ONLY a valid JSON array (no markdown, no extra text):
[
  {{
    "day_number": 1,
    "morning": {{
      "activity": "Attraction Name",
      "description": "What to do there in 2 sentences",
      "location": "Attraction Name",
      "cost": "$10"
    }},
    "afternoon": {{
      "activity": "Attraction Name",
      "description": "What to do there in 2 sentences",
      "location": "Attraction Name",
      "cost": "$15"
    }},
    "evening": {{
      "activity": "Attraction Name",
      "description": "What to do there in 2 sentences",
      "location": "Attraction Name",
      "cost": "$20"
    }}
  }}
]"""
        else:
            prompt = f"""Create a {days}-day travel itinerary for {destination}.
Budget: ${budget} for {num_people} people.
Interests: {', '.join(interests) if interests else 'general tourism'}
Return ONLY a valid JSON array with day_number, morning, afternoon, evening.
Each slot needs: activity, description, location, cost."""

        raw = call_gemini_with_retry(prompt)
        raw = clean_json_response(raw)
        set_cache(cache_key, raw)

        return jsonify({
            'success': True,
            'itinerary': raw,
            'attractions_used': len(attractions),
            'from_dataset': len(attractions) > 0
        })

    except Exception as e:
        print(f"Trip ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        destination = data.get('destination', '')
        rec_type = data.get('type', 'hotels')
        budget = data.get('budget', 500)
        days = data.get('days', 3)
        members = data.get('members', 1)

        cache_key = f"{rec_type}_{destination}_{budget}"
        cached = get_cached(cache_key)
        if cached:
            print(f"Cache hit: {rec_type} for {destination}")
            return jsonify({'success': True, 'data': cached, 'type': rec_type})

        if rec_type == 'hotels':
            prompt = f"""You are a travel expert. Recommend real hotels for {destination}.
Total accommodation budget: ${budget} for {days} nights (${round(budget/max(days,1))}/night).

Return ONLY a valid JSON array of exactly 6 hotels (2 budget, 2 mid-range, 2 luxury).
Use REAL hotel names that actually exist in {destination}:
[
  {{
    "name": "Real Hotel Name",
    "type": "Hostel/Hotel/Resort/5-Star",
    "tier": "budget",
    "price_per_night": "$30",
    "rating": 4.3,
    "area": "Neighborhood name",
    "description": "One sentence why it is great",
    "highlights": ["Free breakfast", "Near metro"]
  }}
]"""

        elif rec_type == 'food':
            prompt = f"""You are a food expert. Recommend real restaurants in {destination}.
Food budget: ${budget} for {days} days, {members} people.

Return ONLY a valid JSON array of exactly 6 restaurants (2 affordable, 2 mid-range, 2 fine dining).
Use REAL restaurant names that actually exist in {destination}:
[
  {{
    "name": "Real Restaurant Name",
    "cuisine": "Japanese",
    "tier": "affordable",
    "price_per_meal": "$10-15",
    "rating": 4.5,
    "area": "Neighborhood",
    "description": "One sentence about this place",
    "must_try": "Signature dish name",
    "dietary": "Both"
  }}
]"""

        elif rec_type == 'transport':
            prompt = f"""You are a transport expert for {destination}.
Trip: {days} days, {members} people, transport budget ${budget}.

Return ONLY a valid JSON object:
{{
  "getting_there": [
    {{
      "type": "Flight",
      "price_range": "$200-400",
      "duration": "3 hours",
      "provider": "Skyscanner",
      "url": "https://www.skyscanner.com"
    }},
    {{
      "type": "Train",
      "price_range": "$50-100",
      "duration": "6 hours",
      "provider": "Trainline",
      "url": "https://www.thetrainline.com"
    }}
  ],
  "getting_around": [
    {{
      "type": "Metro",
      "cost": "$2 per ride",
      "description": "How to use metro in {destination}",
      "tip": "Useful local tip"
    }},
    {{
      "type": "Taxi/Uber",
      "cost": "$5-15 per ride",
      "description": "Best for late nights",
      "tip": "Use app to avoid overcharging"
    }},
    {{
      "type": "Bus",
      "cost": "$1 per ride",
      "description": "Cheapest option in {destination}",
      "tip": "Buy day pass for savings"
    }}
  ],
  "daily_budget": "$15",
  "best_app": "Best transport app name for {destination}"
}}"""

        elif rec_type == 'activities':
            prompt = f"""You are a travel expert. List 10 must-visit attractions in {destination}.
Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "name": "Attraction Name",
    "type": "Museum",
    "description": "Two sentence description of what makes this place special.",
    "rating": 4.5,
    "cost": 15,
    "isUNESCO": false
  }}
]
Rules:
- cost is a NUMBER in USD (use 0 if free, not null)
- type is one of: Museum, Historical, Nature, Religious, Market, Art, Entertainment, Park
- isUNESCO is true or false only
- rating is between 3.5 and 5.0"""

        else:
            return jsonify({'success': False, 'error': 'Invalid type'}), 400

        raw = call_gemini_with_retry(prompt)
        raw = clean_json_response(raw)
        set_cache(cache_key, raw)

        return jsonify({'success': True, 'data': raw, 'type': rec_type})

    except Exception as e:
        print(f"Recommendations ERROR: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'message': 'Backend is alive!',
        'cached_items': len(recommendations_cache),
        'destinations_loaded': len(ALL_DESTINATIONS),
        'ml_models': ['tfidf_cosine_similarity', 'popularity_scoring', 'linear_regression_budget']
    })


# ════════════════════════════════════════════════════════════════
# ML ENDPOINT 1: Content-Based Recommendations (TF-IDF + Cosine)
# ════════════════════════════════════════════════════════════════
@app.route('/api/recommend', methods=['POST'])
def ml_recommend():
    try:
        data = request.json or {}
        past = data.get("past_destinations", [])

        if not past or TFIDF_MATRIX is None:
            # Fallback: top rated
            fallback = sorted(ALL_DESTINATIONS,
                key=lambda x: x.get("rating", 0), reverse=True)[:10]
            return jsonify({
                "recommendations": fallback,
                "algorithm": "top_rated_fallback"
            })

        query_text = " ".join(past)
        query_vec = TFIDF_VECTORIZER.transform([query_text])
        scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

        past_lower = [p.lower() for p in past]
        top_indices = scores.argsort()[::-1]

        results = []
        for idx in top_indices:
            dest = ALL_DESTINATIONS[idx]
            if dest.get("city", "").lower() not in past_lower:
                results.append(dest)
            if len(results) >= 10:
                break

        return jsonify({
            "recommendations": results,
            "algorithm": "tfidf_cosine_similarity",
            "based_on": past
        })

    except Exception as e:
        print(f"Recommend ERROR: {e}")
        return jsonify({"error": str(e), "recommendations": []}), 500


# ════════════════════════════════════════════════════════════════
# ML ENDPOINT 2: Trending (Popularity Scoring)
# ════════════════════════════════════════════════════════════════
@app.route('/api/trending', methods=['GET'])
def ml_trending():
    try:
        return jsonify({
            "trending": TRENDING_CACHE,
            "algorithm": "weighted_popularity_score",
            "formula": "rating * log(1 + totalAttractions)",
            "count": len(TRENDING_CACHE)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trending": []}), 500


# ════════════════════════════════════════════════════════════════
# ML ENDPOINT 3: Budget Predictor (Linear Regression)
# ════════════════════════════════════════════════════════════════
@app.route('/api/predict-budget', methods=['POST'])
def ml_predict_budget():
    try:
        data = request.json or {}
        destination = data.get("destination", "").lower()
        days = max(1, int(data.get("days", 3)))
        members = max(1, int(data.get("members", 1)))

        is_expensive = 1 if any(c in destination for c in EXPENSIVE_CITIES) else 0
        is_europe = 1 if any(c in destination for c in EUROPE_COUNTRIES) else 0
        is_asia = 1 if any(c in destination for c in ASIA_COUNTRIES) else 0

        features = [[days, members, is_expensive, is_europe, is_asia]]
        predicted = max(100, round(float(BUDGET_MODEL.predict(features)[0])))

        low = round(predicted * 0.8)
        high = round(predicted * 1.2)

        ppd = predicted / (days * members)
        if ppd < 30: tier = "Budget"
        elif ppd < 100: tier = "Mid-Range"
        elif ppd < 250: tier = "Comfort"
        else: tier = "Luxury"

        return jsonify({
            "destination": data.get("destination", ""),
            "predicted_budget": predicted,
            "range": {"low": low, "high": high},
            "per_person_per_day": round(ppd),
            "tier": tier,
            "algorithm": "linear_regression",
            "r_squared": round(float(BUDGET_MODEL.score(
                [[days, members, is_expensive, is_europe, is_asia]],
                [predicted]
            )), 3),
            "features_used": {
                "days": days,
                "members": members,
                "is_expensive_city": bool(is_expensive),
                "is_europe": bool(is_europe),
                "is_asia": bool(is_asia)
            }
        })

    except Exception as e:
        print(f"Budget predict ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
