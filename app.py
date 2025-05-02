from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import feedparser
import random
import requests
from bs4 import BeautifulSoup
import re
import time
import json

# Replace with your Seedr credentials
SEEDR_EMAIL = "artificialintelligenceee2025@gmail.com"
SEEDR_PASSWORD = "artificialintelligenceee2025@"

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

#
movie_cache = []


# Input model
class Query(BaseModel):
    query: str

# Simpler version without logger dependency
def extract_magnet_links_with_size(forum_url):
    try:
        response = requests.get(forum_url, timeout=10)  # Add timeout
        if response.status_code != 200:
            print(f"Failed to fetch URL: {forum_url}, status: {response.status_code}")
            return []  # Return empty list instead of invalid placeholders

        soup = BeautifulSoup(response.content, 'html.parser')
        links_with_size = []

        # Search for magnet links
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('magnet:?'):
                size = "Unknown"
                
                # First try to extract size from the magnet link's dn parameter
                if "&dn=" in href:
                    dn_part = href.split("&dn=")[1].split("&")[0]
                    # URL decode the dn parameter
                    import urllib.parse
                    dn_decoded = urllib.parse.unquote(dn_part)
                    
                    # Look for specific size patterns in the filename
                    size_patterns = [
                        r'(\d+\.\d+GB)',
                        r'(\d+GB)',
                        r'(\d+\.\d+MB)',
                        r'(\d+MB)'
                    ]
                    
                    for pattern in size_patterns:
                        size_match = re.search(pattern, dn_decoded, re.IGNORECASE)
                        if size_match:
                            size = size_match.group(1)
                            break
                    
                    # If size wasn't found with exact pattern, look for size with space
                    if size == "Unknown":
                        space_patterns = [
                            r'(\d+\.\d+\s*GB)',
                            r'(\d+\s*GB)',
                            r'(\d+\.\d+\s*MB)',
                            r'(\d+\s*MB)'
                        ]
                        for pattern in space_patterns:
                            size_match = re.search(pattern, dn_decoded, re.IGNORECASE)
                            if size_match:
                                size = size_match.group(1).replace(" ", "")
                                break
                
                # If still unknown, try to find near the link
                if size == "Unknown":
                    # Try surrounding text
                    parent_text = a.parent.get_text(" ", strip=True) if a.parent else ""
                    for pattern in [r'(\d+\.\d+\s*GB)', r'(\d+\s*GB)', r'(\d+\.\d+\s*MB)', r'(\d+\s*MB)']:
                        size_match = re.search(pattern, parent_text, re.IGNORECASE)
                        if size_match:
                            size = size_match.group(1).replace(" ", "")
                            break
                
                # If we still haven't found the size, try one more approach by analyzing link text
                if size == "Unknown" and a.text:
                    link_text = a.text.strip()
                    for pattern in [r'(\d+\.\d+\s*GB)', r'(\d+\s*GB)', r'(\d+\.\d+\s*MB)', r'(\d+\s*MB)']:
                        size_match = re.search(pattern, link_text, re.IGNORECASE)
                        if size_match:
                            size = size_match.group(1).replace(" ", "")
                            break

                # Special case for this specific feed: extract size from file name format
                if size == "Unknown" and "&dn=" in href:
                    # Look explicitly for the common pattern in your magnet links
                    size_pattern = r'[-\s](\d+\.\d+GB|'\
                                  r'\d+GB|'\
                                  r'\d+\.\d+MB|'\
                                  r'\d+MB)[\.\s]'
                    size_match = re.search(size_pattern, dn_decoded, re.IGNORECASE)
                    if size_match:
                        size = size_match.group(1)

                # Only add entries where we could determine a valid size
                if size != "Unknown" and re.match(r"^\d+(\.\d+)?(MB|GB)$", size, re.IGNORECASE):
                    links_with_size.append({
                        "size": size,
                        "magnet": href
                    })

        return links_with_size

    except Exception as e:
        print(f"Error extracting magnet links: {e}")
        return []  # Return empty list on error

# /search API with improved error handling (using print instead of logger)

RSS_FEEDS = [
    "https://rss.app/feeds/Mm5QLqlnzIk7CKPn.xml",
    "https://rss.app/feeds/YoxoeKAYrY0jJvof.xml",
]

# Function to fetch and combine data from multiple RSS feeds
def fetch_movies_from_feeds():
    movie_cache = []
    for rss_url in RSS_FEEDS:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            # Extract movie details
            movie_data = {
                "title": entry.title,
                "summary": entry.summary,
                "link": entry.link,
                "image": None,  # Default to None
                "rating": None  # Default to None
            }

            # Extract image (check for <media:content> or <enclosure>)
            if 'media_content' in entry:
                # If media content is available, we might find an image here
                movie_data['image'] = entry.media_content[0]['url']
            elif 'enclosures' in entry:
                # Sometimes, an image might be included in the enclosure tag
                for enclosure in entry.enclosures:
                    if enclosure.type.startswith('image'):
                        movie_data['image'] = enclosure.href

            # Extract rating (if available)
            if 'rating' in entry:
                movie_data['rating'] = entry.rating  # Example: might be in <rating> tag

            movie_cache.append(movie_data)
    return movie_cache

@app.post("/search")
async def search_movie(query: Query):
    try:
        # Fetch the movies from the combined RSS feeds
        movie_cache = fetch_movies_from_feeds()

        if not query.query:
            return JSONResponse(
                status_code=400,
                content={"error": "No query provided", "success": False}
            )

        if not movie_cache:
            return JSONResponse(
                status_code=404,
                content={"error": "No movies in feed", "success": False}
            )

        movie_texts = [m["title"] + " " + m["summary"] for m in movie_cache]
        movie_vectors = model.encode(movie_texts)  # Ensure your model is defined
        query_vector = model.encode([query.query])

        # Compute cosine similarity
        sims = cosine_similarity(query_vector, movie_vectors)[0]
        best_idx = int(np.argmax(sims))
        match = movie_cache[best_idx]
        match_score = float(sims[best_idx])

        # Skip low confidence matches
        if match_score < 0.1:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "No good matches found for your query",
                    "success": False
                }
            )

        # Get magnet links with size
        files = extract_magnet_links_with_size(match["link"])

        # Return response with valid files or no files found
        if not files:
            return {
                "title": match["title"],
                "score": match_score * 100,
                "link": match.get("link", ""),
                "image": match.get("image", ""),
                "rating": match.get("rating", "N/A"),
                "files": [],
                "success": True,
                "message": "🎥 Found it! 😔 Sadly, no working download links. Please try again"
            }

        return {
            "title": match["title"],
            "score": match_score * 100,
            "files": files,
            "link": match.get("link", ""),
            "image": match.get("image", ""),
            "rating": match.get("rating", "N/A"),
            "success": True
        }

    except Exception as e:
        print(f"Search error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error while processing request",
                "success": False
            }
        )
# Add these imports at the top of your file if not already present
# Function to check account space
def check_account_space(token):
    url = f"https://www.seedr.cc/api/folder?access_token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        space_max = data.get("space_max", 0)
        space_used = data.get("space_used", 0)
        return {
            "space_used": space_used,
            "space_max": space_max,
            "space_available": space_max - space_used,
            "percent_used": (space_used / space_max * 100) if space_max > 0 else 0
        }
    return None

# Function to get wishlist items
def get_wishlist(token):
    url = f"https://www.seedr.cc/api/wishlist?access_token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("wishlist", [])
    return []

# Function to get Seedr access token
def get_seedr_token(email, password):
    url = "https://www.seedr.cc/oauth_test/token.php"
    data = {
        'grant_type': 'password',
        'client_id': 'seedr_chrome',
        'type': 'login',
        'username': email,
        'password': password
    }
    
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Failed to get Seedr token: {response.text}")

# Function to check if magnet is already in Seedr
def check_existing_folders(token):
    url = f"https://www.seedr.cc/api/folder?access_token={token}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("folders", [])
    return []

# Function to get folder contents
def get_folder_contents(token, folder_id=None):
    if folder_id:
        url = f"https://www.seedr.cc/api/folder/{folder_id}?access_token={token}"
    else:
        url = f"https://www.seedr.cc/api/folder?access_token={token}"
    
    response = requests.get(url)
    return response.json()

# Function to get file download link
def get_file_link(token, file_id):
    url = "https://www.seedr.cc/oauth_test/resource.php"
    data = {
        'access_token': token,
        'func': 'fetch_file',
        'folder_file_id': file_id
    }
    
    response = requests.post(url, data=data)
    return response.json()

# Function to add magnet link to Seedr
def add_magnet_to_seedr(token, magnet_link):
    url = "https://www.seedr.cc/oauth_test/resource.php"
    data = {
        'access_token': token,
        'func': 'add_torrent',
        'torrent_magnet': magnet_link
    }
    
    response = requests.post(url, data=data)
    return response.json()

# Model for magnet request
class MagnetRequest(BaseModel):
    magnet: str

@app.post("/seedr-download")
async def seedr_download(request: MagnetRequest):
    try:
        # Get token
        token = get_seedr_token(SEEDR_EMAIL, SEEDR_PASSWORD)
        
        # Check available space and get existing folders before proceeding
        space_info = check_account_space(token)
        if not space_info:
            return {"status": "error", "message": "Failed to get account space information"}
        
        # Get existing folders BEFORE adding magnet
        existing_folders = check_existing_folders(token)
        
        # Add magnet link
        try:
            add_result = add_magnet_to_seedr(token, request.magnet)
            print(f"Add magnet result: {add_result}")  # Debug log
            
            # Check if Seedr returned a space-related error
            if add_result.get("result") == "not_enough_space_added_to_wishlist" or "not_enough_space" in str(add_result):
                return {
                    "status": "error", 
                    "message": "Not enough space available. Please upgrade your plan.",
                    "space_info": space_info,
                    "folders_count": len(existing_folders),
                    "folders": [{"id": f["id"], "name": f["name"], "size": f["size"]} for f in existing_folders]
                }
        except Exception as e:
            print(f"Error during add_magnet_to_seedr: {str(e)}")
            # Even if there's an error, continue with the process, as the magnet might have been added
        
        # Wait a bit for Seedr to process the magnet
        time.sleep(5)  # Increased wait time to 5 seconds
        
        # Get updated folders AFTER adding magnet
        new_folders = check_existing_folders(token)
        
        # First check if there are any new folders (comparing before and after)
        added_folders = []
        for folder in new_folders:
            if not any(ef["id"] == folder["id"] for ef in existing_folders):
                added_folders.append(folder)
        
        # If we found newly added folders
        if added_folders:
            folder = added_folders[0]  # Take the first new folder
            folder_id = folder["id"]
            
            # Wait for download to complete (max 60 seconds)
            for _ in range(60):
                # Get folder contents
                folder_contents = get_folder_contents(token, folder_id)
                
                # Check if we have files in the folder
                if folder_contents.get("files"):
                    # Get first file
                    file_id = folder_contents["files"][0]["folder_file_id"]
                    
                    # Get download link
                    download_info = get_file_link(token, file_id)
                    
                    # Return the download link
                    if "url" in download_info:
                        return {
                            "status": "success",
                            "download_url": download_info["url"],
                            "file_name": folder_contents["files"][0]["name"],
                            "file_size": folder_contents["files"][0]["size"]
                        }
                time.sleep(1)
        
        # If no new folders were added OR we couldn't get a download link from the new folder,
        # check ALL folders (not just new ones) for available files
        for folder in new_folders:
            try:
                folder_contents = get_folder_contents(token, folder["id"])
                
                # Check if we have files in the folder
                if folder_contents.get("files"):
                    # Get first file
                    file_id = folder_contents["files"][0]["folder_file_id"]
                    
                    # Get download link
                    download_info = get_file_link(token, file_id)
                    
                    # Return the download link
                    if "url" in download_info:
                        return {
                            "status": "success",
                            "download_url": download_info["url"],
                            "file_name": folder_contents["files"][0]["name"],
                            "file_size": folder_contents["files"][0]["size"]
                        }
            except Exception as e:
                print(f"Error checking folder {folder['id']}: {str(e)}")
                continue
        
        # If we get here, something went wrong - check if there are any folders at all
        if new_folders:
            return {
                "status": "error", 
                "message": "Files found but could not get download links. Try accessing files directly from seedr.cc",
                "folders_count": len(new_folders),
                "folders": [{"id": f["id"], "name": f["name"], "size": f["size"]} for f in new_folders]
            }
        else:
            return {
                "status": "error", 
                "message": "Could not find any files. There may be an issue with the magnet link.",
                "space_info": space_info,
                "folders_count": 0,
                "folders": []
            }
            
    except Exception as e:
        print(f"Main exception in seedr_download: {str(e)}")
        return {"status": "error", "message": str(e)}
        
# Add a new endpoint to get account status
@app.get("/seedr-status")
async def seedr_status():
    try:
        token = get_seedr_token(SEEDR_EMAIL, SEEDR_PASSWORD)
        space_info = check_account_space(token)
        wishlist = get_wishlist(token)
        folders = get_folder_contents(token).get("folders", [])
        
        return {
            "status": "success",
            "space_info": space_info,
            "wishlist_count": len(wishlist),
            "wishlist": wishlist,
            "folders_count": len(folders),
            "folders": [{"id": f["id"], "name": f["name"], "size": f["size"]} for f in folders]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/seedr-download/{folder_id}")
async def get_direct_download_url(folder_id: int):
    try:
        # Get token
        token = get_seedr_token(SEEDR_EMAIL, SEEDR_PASSWORD)
        
        # Get folder contents
        folder_contents = get_folder_contents(token, folder_id)
        
        # Check if we have files in the folder
        if folder_contents.get("files"):
            # Get first file
            file_id = folder_contents["files"][0]["folder_file_id"]
            
            # Get download link
            download_info = get_file_link(token, file_id)
            
            # Return the download link
            if "url" in download_info:
                return {
                    "status": "success",
                    "download_url": download_info["url"],
                    "file_name": folder_contents["files"][0]["name"],
                    "file_size": folder_contents["files"][0]["size"]
                }
        
        return {
            "status": "error",
            "message": "No downloadable files found in the specified folder"
        }
            
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Function to delete a folder in Seedr
def delete_seedr_folder(token, folder_id):
    url = "https://www.seedr.cc/oauth_test/resource.php"
    data = {
        'access_token': token,
        'func': 'delete',
        'delete_arr': json.dumps([{"type": "folder", "id": folder_id}])
    }
    
    response = requests.post(url, data=data)
    return response.json()

# API endpoint to delete a folder
@app.delete("/seedr-folder/{folder_id}")
async def delete_folder(folder_id: int):
    try:
        # Get token
        token = get_seedr_token(SEEDR_EMAIL, SEEDR_PASSWORD)
        
        # Delete the folder
        delete_result = delete_seedr_folder(token, folder_id)
        
        # Check if deletion was successful
        if delete_result.get("result") == "success":
            # Get updated folder list after deletion
            current_folders = check_existing_folders(token)
            
            return {
                "status": "success",
                "message": f"Folder {folder_id} deleted successfully",
                "folders_count": len(current_folders),
                "folders": [{"id": f["id"], "name": f["name"], "size": f["size"]} for f in current_folders]
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to delete folder: {delete_result.get('error', 'Unknown error')}",
                "error_details": delete_result
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
# @app.get("/refresh-feed")
# async def refresh_feed():
#     global movie_cache
#     movie_cache = fetch_rss_movies()
#     return {"status": "refreshed", "total": len(movie_cache)}

# # Get random movie
# @app.get("/random")
# async def random_movie():
#     if not movie_cache:
#         return {"error": "Feed is empty"}
#     movie = random.choice(movie_cache)
#     return {
#         "title": movie["title"],
#         "link": movie["link"]
#     }

# # Placeholder for top-rated
# @app.get("/top-rated")
# async def top_rated():
#     return {
#         "note": "Top-rated movies feature coming soon!"
#     }
