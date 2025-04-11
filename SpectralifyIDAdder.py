import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import os
import json
import requests.exceptions
from tqdm import tqdm
import backoff

# Gets connected to Spotify
@backoff.on_exception(backoff.expo, 
                     (requests.exceptions.RequestException, 
                      spotipy.exceptions.SpotifyException),
                     max_tries=5)

def search_spotify(sp, query):
    """Search Spotify with exponential backoff for network issues"""
    return sp.search(q=query, type='track', limit=1)

def add_spotify_ids(input_csv, output_csv, client_id, client_secret):
    """
    Adds Spotify track IDs to a CSV file of songs. Resilient to connection issues.
    Places the track_id as the first column in the output CSV.
    
    # Create a checkpoint file path
    checkpoint_file = f"{output_csv}.checkpoint.json"
    
    # Set up Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, 
        client_secret=client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Load progress from checkpoint if it exists
    processed_tracks = {}
    last_processed_index = -1
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_tracks = checkpoint_data.get('processed_tracks', {})
                last_processed_index = checkpoint_data.get('last_index', -1)
                print(f"Loaded checkpoint: {len(processed_tracks)} tracks already processed")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    # Create the track_id column if it doesn't exist
    if 'track_id' not in df.columns:
        df['track_id'] = None
    
    # Restore progress from checkpoint
    for track_key, track_id in processed_tracks.items():
        title, artist = track_key.split(" ||| ")
        mask = (df['Title'] == title) & (df['Artist'] == artist)
        df.loc[mask, 'track_id'] = track_id
    
    # Process each row with progress bar
    print(f"Processing tracks starting from index {last_processed_index + 1}")
    for index, row in tqdm(list(df.iloc[last_processed_index + 1:].iterrows())):
        # Skip if we already have an ID
        if pd.notna(row.get('track_id')):
            continue
            
        try:
            # Get the track name and artist
            track_name = row['Title']
            artist_name = row['Artist']
            
            # Clean up artist name by taking first artist if there are multiple
            if ';' in artist_name:
                artist_name = artist_name.split(';')[0].strip()
            
            # Handle the case where Charli XCX is written in different case formats
            if 'charli xcx' in artist_name.lower():
                artist_name = 'Charli XCX'
                
            print(f"Searching for: {track_name} by {artist_name}")
            
            # Search for the track with backoff for network issues
            query = f"track:{track_name} artist:{artist_name}"
            results = search_spotify(sp, query)
            
            # Check if we have results
            if results['tracks']['items']:
                track_id = results['tracks']['items'][0]['id']
                df.at[index, 'track_id'] = track_id
                
                # Save to checkpoint
                track_key = f"{track_name} ||| {row['Artist']}"
                processed_tracks[track_key] = track_id
                
                # Update checkpoint file after each successful lookup
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed_tracks': processed_tracks,
                        'last_index': index
                    }, f)
                
                print(f"Found ID: {track_id}")
            else:
                print(f"No results found for {track_name} by {artist_name}")
                
                # Still mark as processed in checkpoint
                track_key = f"{track_name} ||| {row['Artist']}"
                processed_tracks[track_key] = "NOT_FOUND"
                
                # Update checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed_tracks': processed_tracks,
                        'last_index': index
                    }, f)
                
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
                
        except Exception as e:
            print(f"Error processing {track_name} by {artist_name}: {str(e)}")
            
            # Save current progress before potentially failing
            # Reorganize columns with track_id as the first column for the intermediate save
            cols = df.columns.tolist()
            track_id_index = cols.index('track_id')
            cols.pop(track_id_index)
            cols = ['track_id'] + cols
            df_output = df[cols]
            df_output.to_csv(output_csv, index=False)
            print(f"Saved intermediate progress to {output_csv}")
            
            # Update checkpoint with last successful index
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed_tracks': processed_tracks,
                    'last_index': index - 1  # Last successful
                }, f)
    
    # Reorganize columns with track_id as the first column
    cols = df.columns.tolist()
    track_id_index = cols.index('track_id')
    cols.pop(track_id_index)
    cols = ['track_id'] + cols
    df_output = df[cols]
    
    # Save the final updated DataFrame
    df_output.to_csv(output_csv, index=False)
    
    # Remove checkpoint if everything completed successfully
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        
    print(f"Saved output to {output_csv}")
    
    # Print summary
    found_count = df['track_id'].notna().sum()
    total_count = len(df)
    print(f"Found Spotify IDs for {found_count} out of {total_count} tracks ({found_count/total_count*100:.2f}%)")

if __name__ == "__main__":
    # Replace with your Spotify API credentials from the Spotify Dashboard
    CLIENT_ID = "your_client_id"
    CLIENT_SECRET = "yout_client_secret"
    
    # Replace with your file paths
    INPUT_CSV = "data.csv"  # Your input CSV file
    OUTPUT_CSV = "Spectralify2Spotify.csv"  # Output file name
    
    try:
        add_spotify_ids(INPUT_CSV, OUTPUT_CSV, CLIENT_ID, CLIENT_SECRET)
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Progress has been saved to checkpoint.")
        print("Run the script again to resume from where you left off.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Progress has been saved to checkpoint. Run the script again to resume.")