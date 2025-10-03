import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image


def fclip_prediction(images, texts, fclip):
    image_embeddings = fclip.encode_images(images, batch_size=32)
    text_embeddings = fclip.encode_text(texts, batch_size=32)

    # Normalize embeddings to unit norm
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)

    # Compute similarity: text_embeddings dot image_embeddings.T
    similarity_matrix = text_embeddings.dot(image_embeddings.T)  # shape: (num_texts, num_images)

    # For each image, get the index of the most similar text
    best_indices = np.argmax(similarity_matrix, axis=0)  # shape: (num_images,)

    # Get predicted texts and their similarity scores
    predicted = [texts[i] for i in best_indices]
    best_scores = [similarity_matrix[i, j] for j, i in enumerate(best_indices)]

    return predicted


if __name__ == "__main__":
    # Load images and texts
    folder_path = "/Users/tracieluong/Documents/runway-images/ralph-lauren/spring-2026-ready-to-wear"
    valid_extensions = (".png", ".jpeg", ".jpg")
    images = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_extensions)
]
    texts = [
        "linen blazer",
        "satin cocktail dress",
        "trench coat",
        "pleated trousers",
        "wide-leg trousers",
        "straight-leg trousers",
        "cropped trousers",
        "flared trousers",
        "tapered trousers",
        "culottes",
        "paperbag waist pants",
        "high-waist pants",
        "palazzo pants",
        "skinny pants",
        "knit dress",
        "pinstripe suit",
        "chiffon skirt",
        "embroidered maxi dress",
        "leather jacket",
        "jumpsuit",
        "jeans",
        "wool coat",
        "velvet blazer",
        "silk gown",
        "turtleneck sweater",
        "pencil skirt",
        "wrap dress",
        "denim jacket",
        "cardigan",
        "blazer dress",
        "sheer blouse",
        "maxi dress",
        "shift dress",
        "peacoat",
        "blouse",
        "tank top",
        "cape coat",
        "oversized sweater",
        "cropped jacket",
        "bodycon dress",
        "halter top",
        "sweatshirt",
        "parka",
        "bohemian dress",
        "tunic",
        "crop top",
        "polo shirt",
        "culotte jumpsuit",
        "skater dress",
        "ruffle skirt",
        "trouser suit",
        "kaftan",
        "kimono",
        "bomber jacket",
        "asymmetric dress",
        "peplum top",
        "cape dress",
        "mini skirt",
        "oversized coat",
        "longline blazer",
        "ruffled blouse",
        "button-up shirt",
        "wrap jumpsuit",
        "sweater dress",
        "trench dress",
        "structured coat",
        "slip dress",
        "floor-length gown",
        "halter dress",
        "tank dress",
        "wide-leg jumpsuit",
        "bodysuit",
        "poncho",
        "sheath dress",
        "ruffle dress",
        "puff-sleeve blouse",
        "wrap top",
        "pleated pants",
        "suit jacket",
        "cap sleeve top",
        "long-sleeve blouse",
        "denim skirt",
        "suit vest",
        "wrap coat",
        "structured blazer",
        "bustier top",
        "cutout dress",
        "shift jumpsuit",
        "lace dress",
        "fringe jacket",
        "kimono coat",
        "blazer vest",
        "tie-front blouse",
        "asymmetric skirt",
        "cape jacket",
        "maxi coat",
        "off-shoulder dress",
        "corset top",
        "peplum dress",
        "ball gown",
        "pleated blouse",
        "puff-sleeve dress",
        "shirt dress", 
        "sun dress",
        "sombrero"
    ]

    colors = [
    # Basic colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray",

    # Neutrals / earth tones
    "beige", "tan", "cream", "ivory", "taupe", "khaki", "olive", "sand", "charcoal", "camel", "off-white",

    # Pastels
    "baby blue", "mint green", "lavender", "peach", "light pink", "pale yellow", "powder blue",

    # Dark / jewel tones
    "navy", "maroon", "burgundy", "forest green", "emerald", "teal", "plum", "sapphire",

    # Metallics / shiny
    "gold", "silver", "bronze", "copper", "rose gold",

    # Others / vibrant
    "magenta", "turquoise", "lime green", "chartreuse", "coral", "cobalt", "mustard", "fuchsia",

    # Patterns / multi-color
    "floral", "striped", "polka dot", "plaid", "paisley", "camouflage", "tie-dye"
]

    # Initialize FashionCLIP model
    fclip = FashionCLIP('fashion-clip')

    # --- Predict once for all images ---
    predicted_articles = fclip_prediction(images, texts, fclip)
    # --- Predict colors ---
    predicted_colors = fclip_prediction(images, colors, fclip)

    # --- Create a table ---
    df = pd.DataFrame({
        "image_file": [os.path.basename(p) for p in images],
        "article": predicted_articles,
        # "article_score": best_scores,
        "color": predicted_colors,
        # "color_score": color_scores
    })

    # --- Print table ---
    df.to_csv("fashionclip_predictions.csv", index=False)

    # Show first 6 predictions with images
    n_show = 6
    subset = df.head(n_show)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows x 3 cols
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, subset.iterrows()):
        img_path = os.path.join(folder_path, row["image_file"])
        img = Image.open(img_path)

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{row['article']} ({row['color']})\n",
                    # f"Score: {row['article_score']:.2f}, {row['color_score']:.2f}",
                    fontsize=9)

    plt.tight_layout()
    plt.show()