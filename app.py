import streamlit as st
import re
from PIL import Image
from transformers import pipeline
from deep_translator import GoogleTranslator
from urllib.parse import quote

st.title("料理認識・栄養情報システム")

FOOD_THRESHOLD = 60
GENERAL_THRESHOLD = 60

@st.cache_resource
def load_models():
    food_classifier = pipeline(
        "image-classification",
        model="nateraw/food"
    )

    general_classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

    return food_classifier, general_classifier

food_classifier, general_classifier = load_models()

special_name_mapping = {
    "Granny Smith": "りんご",
    "Red Delicious": "りんご",
    "coffee mug": "コーヒー",
    "water bottle": "水"
}

label_normalization = {
    "Granny Smith": "apple",
    "Red Delicious": "apple",
    "coffee mug": "coffee",
    "water bottle": "water"
}

category_keywords = {
    "飲み物": [
        "coffee", "espresso", "water", "juice", "tea", "cola", "soda",
        "milk", "latte", "cappuccino", "beer", "wine"
    ],
    "果物": [
        "apple", "banana", "orange", "lemon", "pineapple", "strawberry",
        "grape", "pear", "mango", "kiwi", "watermelon", "melon",
        "blueberry", "raspberry", "cherry", "peach", "plum", "coconut",
        "papaya", "avocado", "fig", "pomegranate"
    ],
    "デザート": [
        "cake", "donut", "doughnut", "ice cream", "chocolate",
        "cookie", "waffle", "pudding", "dessert", "candy", "sweet",
        "cupcake", "brownie", "macaron", "pie","apple pie", "beignets", "bread pudding",
        "cannoli", "carrot cake","cheesecake","chocolate cake","chocolate mousse",
        "churros", "cup cakes","donuts","frozen yogurt","macarons","red velvet cake",
        "strawberry shortcake","tiramisu", "waffles"
    ],
    "ファストフード": [
        "french fries", "hamburger", "hot dog", "fried chicken", "onion rings",
        "pizza", "burger","cheeseburger","double burger","fries","chicken wings","nuggets",
        "chicken nuggets","sandwich","club sandwich","submarine sandwich","burrito","tacos",
        "quesadilla","nachos","chow mein","instant noodles","wrap","shawarma",
        "kebab","grilled cheese sandwich","pulled pork sandwich"
    ],
    "主食": [
        "rice", "noodle", "ramen", "spaghetti", "pasta", "bread",
        "toast", "bagel", "burrito", "taco", "bibimbap", "breakfast burrito",
        "fried rice","garlic bread","lasagna","macaroni and cheese", "pad thai",
        "paella", "pho","risotto","spaghetti bolognese","spaghetti carbonara"

    ],
    "たんぱく質食品": [
        "beef", "steak", "chicken", "pork", "meat", "fish", "egg",
        "salmon", "tuna", "shrimp", "lobster", "crab", "sausage",
        "bacon", "ham", "sushi","baby back ribs", "beef carpaccio",
        "beef tartare","chicken curry","chicken quesadilla","ceviche",
        "crab cakes","eggs benedict","escargots","filet mignon","fish and chips",
        "foie gras","fried calamari","grilled salmon","lobster bisque","lobster roll sandwich",
        "mussels","oysters","pork chop","prime rib","sashimi","seaweed salad","shrimp and grits",
        "tuna tartare","cheese","cheese plate","cheese platter"

    ],
    "野菜料理": [
        "salad", "broccoli", "carrot", "vegetable", "tomato", "cucumber",
        "lettuce", "cabbage", "potato", "onion", "corn", "pepper",
        "mushroom", "spinach", "caesar_salad","caprese_salad","edamame",
        "falafel", "greek_salad"
    ]
}

category_advice = {
    "飲み物": {
        "health": "★★★☆☆",
        "advice": "飲み物の種類によって糖分やカフェイン量が異なるため、摂りすぎに注意しましょう。"
    },
    "果物": {
        "health": "★★★★☆",
        "advice": "果物はビタミンや食物繊維を含みますが、糖分もあるため食べすぎには注意しましょう。"
    },
    "デザート": {
        "health": "★★☆☆☆",
        "advice": "糖分や脂質が多い場合があるため、間食として少量にすることをおすすめします。"
    },
    "ファストフード": {
        "health": "★★☆☆☆",
        "advice": "カロリーや塩分が高くなりやすいため、野菜や飲み物の選び方に注意しましょう。"
    },
    "主食": {
        "health": "★★★☆☆",
        "advice": "炭水化物を多く含むため、野菜やたんぱく質と一緒に食べるとバランスが良くなります。"
    },
    "たんぱく質食品": {
        "health": "★★★☆☆",
        "advice": "たんぱく質を含みますが、脂質や調理方法によってカロリーが高くなる場合があります。"
    },
    "野菜料理": {
        "health": "★★★★☆",
        "advice": "野菜を含む料理は健康的ですが、ドレッシングや油の量に注意しましょう。"
    },
    "未分類": {
        "health": "未評価",
        "advice": "詳細な栄養情報は登録されていません。下の検索ボタンから確認できます。"
    }
}

def normalize_label(label):
    if label in label_normalization:
        return label_normalization[label]
    return label

def infer_category(label):
    normalized_label = normalize_label(label)
    lower_label = normalized_label.lower().replace("_", " ")

    for category, keywords in category_keywords.items():
        for keyword in keywords:
            keyword_lower = keyword.lower().replace("_", " ")

            pattern = r"\b" + re.escape(keyword_lower) + r"\b"

            if re.search(pattern, lower_label):
                return category

    return "未分類"

@st.cache_data
def get_display_name(label):
    if label in special_name_mapping:
        return special_name_mapping[label]

    normalized_label = normalize_label(label)

    if "apple" in normalized_label.lower():
        return "りんご"

    text = normalized_label.replace("_", " ")

    try:
        return GoogleTranslator(
            source="en",
            target="ja"
        ).translate(text)

    except:
        return text

def make_google_search_url(display_name):
    query = f"{display_name} カロリー "
    return f"https://www.google.com/search?q={quote(query)}"

def show_food_info(label, display_name):
    st.subheader("🥗 栄養・健康情報")

    category = infer_category(label)
    health = category_advice[category]["health"]
    advice = category_advice[category]["advice"]

    st.write(f"📂 カテゴリー：{category}")
    st.write(f"⭐ 健康度：{health}")
    st.write(f"💡 アドバイス：{advice}")
    st.write("🔥 参考カロリー：Google検索で確認できます")

    calorie_url = make_google_search_url(display_name)

    st.link_button(
        "🔍 カロリー・栄養情報をGoogleで調べる",
        calorie_url
    )

uploaded_file = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "png", "jpeg"]
)

camera_image = st.camera_input(
    "またはカメラで撮影して認識"
)

image_file = uploaded_file or camera_image

if image_file:
    image = Image.open(image_file).convert("RGB")

    st.image(
        image,
        caption="認識対象の画像",
        use_container_width=True
    )

    food_results = food_classifier(image)
    food_best = food_results[0]
    food_label = food_best["label"]
    food_score = food_best["score"] * 100

    if food_score >= FOOD_THRESHOLD:
        final_label = food_label
        final_score = food_score
        final_results = food_results
        
    else:
        general_results = general_classifier(image)
        general_best = general_results[0]
        general_label = general_best["label"]
        general_score = general_best["score"] * 100

        if general_score >= GENERAL_THRESHOLD:
            final_label = general_label
            final_score = general_score
            final_results = general_results
            
        else:
            final_label = None
            final_score = general_score
            final_results = general_results

    if final_label is None:
        st.error("認識できませんでした。もう一度、明るい場所で食品全体が写るように撮影してください。")

        st.subheader("📊 参考候補")
        for i, result in enumerate(final_results[:3], start=1):
            name = get_display_name(result["label"])
            score = result["score"] * 100
            st.write(f"{i}. {name}：{score:.2f}%")

    else:
        display_name = get_display_name(final_label)

        st.success("認識が完了しました！")
        st.subheader(f"🍽 認識結果：{display_name}")
        st.subheader(f"🎯 信頼度：{final_score:.2f}%")

        st.markdown("---")

        st.subheader("📊 上位3件の候補")
        for i, result in enumerate(final_results[:3], start=1):
            candidate_name = get_display_name(result["label"])
            candidate_score = result["score"] * 100
            st.write(f"{i}. {candidate_name}：{candidate_score:.2f}%")

        st.markdown("---")

        show_food_info(final_label, display_name)

        st.markdown("---")

        google_url = make_google_search_url(display_name)

        if final_score >= 90:
            st.info("AIは高い信頼度でこの食品を認識しました。")
        elif final_score >= 70:
            st.warning("認識結果は参考程度にしてください。")
        else:
            st.error("信頼度がやや低いため、別の角度や明るい場所で撮影すると精度が上がる可能性があります。")

        google_url = f"https://www.google.com/search?q={display_name}+栄養情報"

        st.link_button(
            "🔍 詳しい栄養情報を見る",
            google_url
        )

else:
    st.info("画像をアップロードするか、カメラで撮影してください。")