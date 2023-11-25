from googlesearch import search
from bs4 import BeautifulSoup
import requests
from summary import summarize_text
import time

def google_search(query, num_results=5):
    search_results = list(search(query, num_results=num_results))

    return search_results


def scrape(query):
    try:
        response = requests.get(query)
        soup = BeautifulSoup(response.text, 'html5lib')

        paragraphs = soup.find_all('p')
        text_content = '\n'.join([p.get_text() for p in paragraphs])
        return text_content

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("Too many requests. Waiting and retrying...")
            time.sleep(10)
            return None
        else:
            print(f"HTTP Error: {e}")
            return None
    except Exception as e:
        print(f"Error while scraping {query}: {str(e)}")
        return None


if __name__ == "__main__":

    input_text = input("Enter the text from the book: ")

    search_query = f"{input_text} book summary"
    search_results = google_search(search_query, 10)
    count = 0
    for _, query in enumerate(search_results, start=1):
        result = scrape(query)
        print(result)
        print(f"\nResult {count}: {query}")
        print(summarize_text(result[:130]))
        count += 1
