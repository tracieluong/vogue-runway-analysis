import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from pathlib import Path
from unidecode import unidecode
import csv

# ---------------------------
# Helper Functions
# ---------------------------

def clean_name(name: str) -> str:
    """Convert a designer/show name into URL- and filename-safe string."""
    name = unidecode(name.lower())
    for char in [' ', '.', '&', '+']:
        name = name.replace(char, '-')
    while '--' in name:
        name = name.replace('--', '-')
    return name


def extract_json_from_script(scripts, key_fragment):
    """Extract JSON data from a <script> tag containing a specific key fragment."""
    for script in scripts:
        if script.string and key_fragment in script.string:
            js = script.string
            break
    else:
        return None

    try:
        js_clean = js.split(' = ', 1)[1]
        brace_count = 0
        for i, char in enumerate(js_clean):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    js_clean = js_clean[:i+1]
                    break
        return json.loads(js_clean)
    except Exception as e:
        print(f"❌ JSON extraction failed: {e}")
        return None


def get_best_image_url(sources: dict) -> str:
    """Return the best available image URL from sources dict."""
    for size in ['lg', 'md', 'sm']:
        if size in sources and 'url' in sources[size]:
            return sources[size]['url']
    return None


# ---------------------------
# Designer & Show Scraping
# ---------------------------

def designer_to_shows(designer: str) -> list:
    """Return a list of all shows for a designer."""
    designer_slug = clean_name(designer)
    url = f"https://www.vogue.com/fashion-shows/designer/{designer_slug}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch designer page: {e}")
        return []

    soup = BeautifulSoup(r.content, 'html5lib')
    data = extract_json_from_script(soup.find_all('script', type='text/javascript'), 'window.__PRELOADED_STATE__')
    if not data:
        print("❌ Could not find JSON script")
        return []

    try:
        return [show['hed'] for show in data['transformed']['runwayDesignerContent']['designerCollections']]
    except Exception as e:
        print(f"❌ Failed to parse shows list: {e}")
        return []


def designer_show_to_download_images(designer, show, save_path, include_accessories=False):
    """Download all main + detail images for a show."""
    designer_slug = clean_name(designer)
    show_slug = clean_name(show)
    show_path = Path(save_path) / designer_slug / show_slug
    if show_path.exists():
        print(f"✅ Photos already downloaded for {designer} - {show}")
        return

    url = f"https://www.vogue.com/fashion-shows/{show_slug}/{designer_slug}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch show page: {e}")
        return

    soup = BeautifulSoup(r.content, 'html5lib')
    data = extract_json_from_script(soup.find_all('script', type='text/javascript'), 'runwayShowGalleries')
    if not data:
        print("❌ JSON script not found")
        return

    galleries = data['transformed']['runwayShowGalleries']['galleries']
    if not include_accessories:
        galleries = [galleries[0]] if galleries else []

    show_path.mkdir(parents=True, exist_ok=True)

    for g_idx, gallery in enumerate(galleries):
        for i, item in enumerate(gallery.get('items', [])):
            img_url = get_best_image_url(item.get('image', {}).get('sources', {}))
            if not img_url:
                continue
            export_path = show_path / f"{designer_slug}-{show_slug}-g{g_idx}-i{i}.png"

            try:
                with requests.get(img_url, stream=True, timeout=10) as response:
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    img.save(export_path)
                print(f"Downloaded {img_url}")
            except Exception as e:
                print(f"⚠️ Error downloading {img_url}: {e}")


def designer_to_download_images(designer, save_path):
    """Download all shows for a designer."""
    shows = designer_to_shows(designer)
    for show in shows:
        print(f"📂 Downloading {designer} - {show}")
        designer_show_to_download_images(designer, show, save_path, include_accessories=False)


# ---------------------------
# CSV Export
# ---------------------------

def designer_show_to_csv(designer, show, save_path=None):
    """Export all images for a show to CSV."""
    designer_slug = clean_name(designer)
    show_slug = clean_name(show)

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        csv_path = Path(save_path) / f"{designer_slug}_{show_slug}.csv"
        if csv_path.exists():
            print(f'CSV already exists: {csv_path}')
            return None

    url = f"https://www.vogue.com/fashion-shows/{show_slug}/{designer_slug}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch show page: {e}")
        return None

    soup = BeautifulSoup(r.content, 'html5lib')
    data = extract_json_from_script(soup.find_all('script', type='text/javascript'), 'runwayShowGalleries')
    if not data:
        print(f"❌ Could not load show: {designer} - {show}")
        return None

    try:
        items = data['transformed']['runwayShowGalleries']['galleries'][0].get('items', [])
    except Exception as e:
        print(f"❌ Failed to find gallery items: {e}")
        return None

    rows = []
    for item in items:
        img_url = get_best_image_url(item.get('image', {}).get('sources', {}))
        if img_url:
            rows.append([designer, show, img_url])

    if save_path and rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['designer', 'show', 'image_url'])
            writer.writerows(rows)
        print(f"✅ CSV saved to {csv_path}")

    return rows


def designer_to_csv(designer, save_path):
    """Export all shows for a designer to a single CSV."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    csv_path = Path(save_path) / f"{clean_name(designer)}_all_shows.csv"
    if csv_path.exists():
        print(f"CSV already exists: {csv_path}")
        return

    shows = designer_to_shows(designer)
    if not shows:
        print(f"No shows found for {designer}")
        return

    all_rows = []
    for show in shows:
        print(f"Scraping {designer} - {show}")
        rows = designer_show_to_csv(designer, show)
        if rows:
            all_rows.extend(rows)

    if all_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['designer', 'show', 'image_url'])
            writer.writerows(all_rows)
        print(f"✅ All shows saved to {csv_path}")
    else:
        print("No images found to write.")


def all_designers_to_csv(txt_path, save_path):
    """Export all designers listed in a text file to a single CSV."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    csv_path = Path(save_path) / "all_designers.csv"

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            designers = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error reading file {txt_path}: {e}")
        return

    if not designers:
        print("No designers found in the file.")
        return

    existing_rows = set()
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.add((row['designer'], row['show']))

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if csv_path.stat().st_size == 0:
            writer.writerow(['designer', 'show', 'image_url'])

        for designer in designers:
            print(f"\n📂 Starting designer: {designer}")
            try:
                shows = designer_to_shows(designer)
                for show in shows:
                    if (designer, show) in existing_rows:
                        print(f"✅ Already scraped: {designer} - {show}")
                        continue

                    print(f"🔍 Scraping: {designer} - {show}")
                    rows = designer_show_to_csv(designer, show)
                    if rows:
                        writer.writerows(rows)
                        f.flush()
                        existing_rows.update((designer, show) for _ in rows)
            except Exception as e:
                print(f"❌ Failed to process {designer}: {e}")


if __name__ == "__main__":
    # Example usage:
    designer = "Ralph Lauren"
    show = "Spring 2026 Ready-to-Wear"
    save_path = "/Users/tracieluong/Documents/runway-images"

    # Return all shows for a specific designer
    shows = designer_to_shows(designer)
    print(f"Shows for {designer}: {shows}")

    # Download images for a specific designer and show
    # designer_show_to_download_images(designer, show, save_path, include_accessories=False)

    # Download images for all shows of a specific designer
    # designer_to_download_images(designer, save_path)

    # Save images URLs to CSV for a specific designer and show
    # designer_show_to_csv(designer, show, save_path)

    # Save images URLs to CSV for all shows of a specific designer
    # designer_to_csv(designer, save_path)

    # Save images URLs to CSV for all designers listed in a text file
    # all_designers_to_csv("designers.txt", save_path)
