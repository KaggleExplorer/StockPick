import json
import urllib
import urllib.error
from urllib.request import urlopen
from pathlib2 import Path


def config_provider(config_file):
    f = Path(config_file)
    if f.is_file():
        with open(f) as fh:
            return json.load(fh)
    return None


def attach_api_key_to_url(url, api_key):
    return f'{url}?apikey={api_key}'


def get_json_from_url(url, decode_format='utf-8'):
    try:
        response = urlopen(url)
        data = response.read().decode(decode_format)
        return json.loads(data)
    except urllib.error.HTTPError as e:
        print(e.__dict__)
    except urllib.error.URLError as e:
        print(e.__dict__)
    except ValueError as e:
        print(e.__dict__)
    except Exception:
        raise Exception(f'error when fetching data from {url}')


def find_in_json(obj, key):
    """
    Scan the json file to find the value of the required key.
    :param obj: json file
    :param key: required key
    :return: value
    """
    arr = []

    def extract(obj, arr, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results



