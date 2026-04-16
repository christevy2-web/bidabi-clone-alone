import asyncio
import aiohttp
import csv
import os
from aiohttp import ClientSession, ClientTimeout

# --- CONFIGURATION ---
API_URL = "https://world.openfoodfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

CATEGORY = "butter"  # Exemple : "sugar", "chocolate", "bread", etc.
TARGET_COUNT = 10
PAGE_SIZE = 50
MAX_PAGES = 10

# On ralentit pour éviter l'erreur 503
MAX_CONCURRENT_REQUESTS = 1
MAX_CONCURRENT_IMAGES = 1

# --- HELPERS ---
def get_best_image(product):
    return (
        product.get("image_url")
        or product.get("image_front_url")
        or product.get("image_small_url")
    )

def is_valid_product(product):
    required = ["_id", "product_name", "categories_tags"]
    if not all(product.get(f) for f in required):
        return False
    return bool(get_best_image(product))

def extract_product_info(product):
    img_url = get_best_image(product)
    # On renvoie les données pour le CSV ET l'URL de l'image
    return [
        product.get("_id"),
        product.get("product_name"),
        ", ".join(product.get("categories_tags", [])),
        product.get("ingredients_text", ""),
        img_url
    ]

# --- ASYNC FUNCTIONS ---
async def fetch_page(session, category, page, page_size, sem):
    params = {
        "action": "process",
        "tagtype_0": "categories",
        "tag_contains_0": "contains",
        "tag_0": category,
        "page": page,
        "page_size": page_size,
        "json": 1
    }
    async with sem:
        try:
            async with session.get(API_URL, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("products", [])
                else:
                    print(f"⚠ Erreur {resp.status} sur la page {page}")
                    return []
        except Exception as e:
            print(f"⚠ Erreur API page {page} : {e}")
            return []

async def download_image(session, url, image_id, sem, category_name):
    if not url: return
    
    # Chemin propre vers data/raw/images/sugar
    folder = os.path.join("data", "raw", "images", category_name)
    os.makedirs(folder, exist_ok=True)

    ext = url.split(".")[-1].split("?")[0]
    if len(ext) > 4: ext = "jpg" # Sécurité extension
    filename = os.path.join(folder, f"{image_id}.{ext}")

    if os.path.exists(filename): return

    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(filename, "wb") as f:
                        f.write(content)
                    print(f" Image {image_id} téléchargée.")
        except Exception as e:
            print(f"⚠ Erreur image {image_id} : {e}")

# --- MAIN LOGIC ---
async def scrape(category, target_count, page_size, max_pages):
    timeout = ClientTimeout(total=300)
    sem_api = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    sem_img = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)

    async with ClientSession(headers=HEADERS, timeout=timeout) as session:
        valid_products = []
        image_tasks = []
        page = 1

        while len(valid_products) < target_count and page <= max_pages:
            print(f"→ Recherche page {page}...")
            products = await fetch_page(session, category, page, page_size, sem_api)
            
            if not products:
                break

            for product in products:
                if is_valid_product(product):
                    info = extract_product_info(product)
                    valid_products.append(info)
                    
                    # Lancement du téléchargement
                    image_url = info[4]
                    image_id = info[0]
                    task = asyncio.create_task(download_image(session, image_url, image_id, sem_img, category))
                    image_tasks.append(task)

                    if len(valid_products) >= target_count:
                        break
            page += 1
            await asyncio.sleep(1) # Petite pause pour le serveur

        if image_tasks:
            await asyncio.gather(*image_tasks)
        return valid_products

def save_to_csv(filename, rows):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["foodId", "label", "category", "foodContentsLabel", "image_url"])
        writer.writerows(rows)

def main():
    print(f" Début de la collecte pour : {CATEGORY}")
    products = asyncio.run(scrape(CATEGORY, TARGET_COUNT, PAGE_SIZE, MAX_PAGES))
    
    output_file = f"data/raw/metadata_{CATEGORY}_{len(products)}.csv"
    save_to_csv(output_file, products)
    print(f"\n Terminé ! {len(products)} produits dans {output_file}")

if __name__ == "__main__":
    main()