# db_connection.py
import os
import requests
import pandas as pd

# For Heliohost
PHP_API_BASE_URL = os.environ.get("PHP_API_URL", "https://vynceianoani.helioho.st/skonnect-api")
PHP_API_EVENTS_ENDPOINT = f"{PHP_API_BASE_URL}/getevents.php"

def fetch_events_from_mysql(limit=None):
    """
    Fetch events from the PHP API on Heliohost.
    Returns a DataFrame with flattened sub-events.
    Returns an empty DataFrame if API is unreachable.
    """
    cols = ["id", "title", "description", "date", "time", "location", "image", "status", "created_at", "points", "event_type"]
    
    try:
        # Make GET request to PHP API
        response = requests.get(PHP_API_EVENTS_ENDPOINT, timeout=10, verify=False)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Check if API returned success
        if data.get("status") == "success" and data.get("main_events"):
            # Flatten nested sub_events into a single list
            events_list = []
            for main_event in data.get("main_events", []):
                sub_events = main_event.get("sub_events", [])
                
                # If there are sub_events, add each one to the list
                if sub_events:
                    for sub_event in sub_events:
                        sub_event['main_event_title'] = main_event.get('title', '')
                        sub_event['main_event_id'] = main_event.get('id', '')
                        events_list.append(sub_event)
                else:
                    # If no sub_events, add the main event itself
                    events_list.append({
                        'id': main_event.get('id', ''),
                        'title': main_event.get('title', ''),
                        'description': main_event.get('description', ''),
                        'date': '',
                        'time': '',
                        'location': '',
                        'image': None,
                        'status': 'upcoming',
                        'created_at': main_event.get('created_at', ''),
                        'points': 0,
                        'event_type': 'general'
                    })
            
            # Apply limit if requested
            if limit and len(events_list) > limit:
                events_list = events_list[:limit]
            
            # Convert to DataFrame
            if events_list:
                df = pd.DataFrame(events_list)
                return df
            else:
                return pd.DataFrame(columns=cols)
        else:
            error_msg = data.get('message', 'Unknown error')
            print(f"[db_connection] API returned error: {error_msg}")
            return pd.DataFrame(columns=cols)
            
    except requests.exceptions.Timeout:
        print("[db_connection] PHP API request timed out")
        return pd.DataFrame(columns=cols)
    except requests.exceptions.ConnectionError as e:
        print(f"[db_connection] Failed to connect to PHP API: {e}")
        return pd.DataFrame(columns=cols)
    except requests.exceptions.RequestException as e:
        print(f"[db_connection] API request failed: {e}")
        return pd.DataFrame(columns=cols)
    except ValueError as e:
        print(f"[db_connection] Failed to parse JSON response: {e}")
        return pd.DataFrame(columns=cols)
    except Exception as e:
        print(f"[db_connection] Unexpected error: {e}")
        return pd.DataFrame(columns=cols)
