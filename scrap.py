import requests
import os
from bs4 import BeautifulSoup
import re
import urllib.parse

def download_images(search_term, folder, max_images=100):
    os.makedirs(folder, exist_ok=True)
    base_url = f"https://unsplash.com/s/photos/{urllib.parse.quote(search_term)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {base_url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img', class_=re.compile('czQTa'))  # Unsplash image class (may change)

    downloaded = 0
    for img in img_tags:
        if downloaded >= max_images:
            break
        srcset = img.get('src')
        if not srcset:
            continue
        
        # Extract the first URL from srcset (highest quality available)
        urls = [url.split(' ')[0] for url in srcset.split(',')]
        if not urls:
            continue
        
        img_url = urls[0]  # Use the first URL (can modify to select specific resolution)
        
        # Add quality parameter to get better resolution
        img_url = img_url.split('?')[0] + '?q=80&w=1000'
        
        try:
            img_response = requests.get(img_url, headers=headers, stream=True)
            img_response.raise_for_status()
            file_path = os.path.join(folder, f"image_{downloaded}.jpg")
            with open(file_path, 'wb') as f:
                for chunk in img_response.iter_content(1024):
                    f.write(chunk)
            downloaded += 1
            print(f"Downloaded {downloaded}/{max_images} images to {folder}")
        except requests.RequestException as e:
            print(f"Error downloading {img_url}: {e}")
            continue

    print(f"Finished downloading {downloaded} images for {search_term}")

# Download car and non-car images
download_images("car", "datasets/cars", 100)
download_images("tree", "datasets/trees", 100)
