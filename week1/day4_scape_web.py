import requests

try:
    response = requests.get("https://cbm-binder3-biabkafka02.biab.au.ing.net:9083/connectors", verify=False)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

    connectors = response.json()   # This is a Python list of connector names
    print(f"Found {len(connectors)} connectors")
    for name in connectors:
        print(f"- {name}")

except requests.exceptions.RequestException as e:
    print(f"Error during request to {response.request.url if 'response' in locals() else 'unknown URL'}: {e}")
except Exception as e:
    import traceback
    print(f"An unexpected error occurred in the connector fetching process: {e}")
    traceback.print_exc()
    raise