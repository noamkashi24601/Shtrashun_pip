import urllib.parse
import json
import time
import pandas as pd
import re
import csv
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import requests




def search_nli_simple(search_string: str, retry_count: int = 0):
    """Simple search in NLI API (original functionality)"""
    if retry_count >= len(API_KEYS):
        print(f"All API keys are rate limited. Please try again later.")
        return None

    current_api_key = get_current_api_key()

    try:
        encoded_search = urllib.parse.quote(search_string)
        url = f"{BASE_URL}?api_key={current_api_key}&query=any,contains,{encoded_search}&index=0&sort=rank"

        print(f"Simple search for: {search_string}")
        print(f"URL (first 150 chars): {url[:150]}...")

        response = requests.get(url, timeout=30)

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            try:
                json_response = response.json()
                print(f"JSON response type: {type(json_response)}")
                if isinstance(json_response, dict):
                    print(f"Response keys: {list(json_response.keys())}")
                    if 'docs' in json_response:
                        print(f"Number of docs: {len(json_response['docs'])}")
                return json_response
            except json.JSONDecodeError:
                print("Error: Invalid JSON response")
                print(f"Response text: {response.text[:200]}")
                return None
        elif response.status_code == 429 or "OVER_RATE_LIMIT" in response.text:
            print(f"Rate limit exceeded for key {current_api_key[:5]}...")
            switch_to_next_api_key()
            return search_nli_simple(search_string, retry_count + 1)
        else:
            print(f"Error: {response.status_code} - {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return Noneimport
        requests



# Your API keys
API_KEYS = [
    'Od48KLpdU4sQK4k4d5Lf0rtbcTP1NGpVmJkXurM7',
    'mZcuSKOCiaboR9ctxclfeMACDtXAgoYQL3cdTRWS',
    't2A91mZgPnhShnDsvGWZWmnj9s7OsTDW072xrg3e',
    'm5TELRWYrhqJmVj10VEjUQVPTIduz02QiFpFtWEh'
]

BASE_URL = 'https://api.nli.org.il/openlibrary/search'
current_key_index = 0

# מילון קיצורים לפתיחה
ABBREVIATIONS = {
    'פי׳': 'פירוש',
    'ר׳': 'רב',
    'ה״ר': 'הרב',
    'ע״פ': 'על פי',
    'ס׳': 'ספר'
}


def get_current_api_key():
    return API_KEYS[current_key_index]


def switch_to_next_api_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"Switching to API key: {get_current_api_key()[:5]}...")
    return get_current_api_key()


def expand_abbreviations(text):
    """הרחבת קיצורים בטקסט"""
    if pd.isna(text) or text == '' or str(text).strip() == '':
        return text

    expanded_text = str(text)
    for abbrev, full_form in ABBREVIATIONS.items():
        expanded_text = expanded_text.replace(abbrev, full_form)

    return expanded_text


def clean_book_title(title):
    """ניקוי שם הספר - הסרת "(בלי שער)" והרחבת קיצורים"""
    if pd.isna(title) or title == '' or str(title).strip() == '':
        return None

    cleaned = str(title).strip()

    # הסרת "(בלי שער)"
    cleaned = re.sub(r'\(בלי שער\)', '', cleaned).strip()

    # הרחבת קיצורים
    cleaned = expand_abbreviations(cleaned)

    return cleaned if cleaned else None


def clean_search_term(term):
    """Clean and prepare search term for API"""
    if pd.isna(term) or term == '' or str(term).strip() == '':
        return None

    cleaned = str(term).strip()
    # Remove any problematic characters that might break the API
    cleaned = re.sub(r'["\'\n\r\t]', ' ', cleaned)
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip() if cleaned.strip() else None


def is_valid_date(date_str):
    """בדיקה אם התאריך מוקדם מ-1942"""
    if not date_str:
        return True  # אם אין תאריך, לא לפסול

    # חיפוש שנים בטקסט
    years = re.findall(r'\b(19\d{2}|18\d{2}|17\d{2}|16\d{2})\b', str(date_str))

    for year in years:
        try:
            if int(year) >= 1942:
                return False
        except ValueError:
            continue

    return True


def build_complex_query(search_params: List[Tuple[str, str]]) -> str:
    """
    Build a complex query string for NLI API following the correct format
    search_params: List of tuples (field, value)
    """
    query_parts = []

    for field, value in search_params:
        if value and clean_search_term(value):
            cleaned_value = clean_search_term(value)
            query_parts.append(f"{field},contains,{cleaned_value},AND")

    if not query_parts:
        return None

    query_string = ";".join(query_parts)
    if query_string.endswith(",AND"):
        query_string = query_string[:-4]

    return query_string


def search_nli_complex(query_string: str, retry_count: int = 0):
    """Search the NLI API with complex query and return the response"""
    if retry_count >= len(API_KEYS):
        print(f"All API keys are rate limited. Please try again later.")
        return None

    current_api_key = get_current_api_key()

    try:
        encoded_query = urllib.parse.quote(query_string, safe=',;')
        # הסרת הגבלת bulkSize כדי לקבל את כל התוצאות
        url = f"{BASE_URL}?api_key={current_api_key}&query={encoded_query}&index=0&sort=rank"

        print(f"Complex query: {query_string}")
        print(f"Full URL: {url}")

        response = requests.get(url, timeout=30)

        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response text: {response.text[:200]}")

        if response.status_code == 200:
            try:
                json_response = response.json()
                print(f"JSON response type: {type(json_response)}")
                if isinstance(json_response, dict):
                    print(f"Response keys: {list(json_response.keys())}")
                    if 'docs' in json_response:
                        print(f"Number of docs: {len(json_response['docs'])}")
                elif isinstance(json_response, list):
                    print(f"Response is a list with {len(json_response)} items")
                return json_response
            except json.JSONDecodeError:
                print("Error: Invalid JSON response")
                return None
        elif response.status_code == 429 or "OVER_RATE_LIMIT" in response.text:
            print(f"Rate limit exceeded for key {current_api_key[:5]}...")
            switch_to_next_api_key()
            return search_nli_complex(query_string, retry_count + 1)
        else:
            print(f"Error: {response.status_code} - {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


def normalize_text_for_comparison(text):
    """Normalize text for comparison by removing common variations"""
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r'[״׳\'".,\-\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_api_fields(item):
    """Extract relevant fields from API response item for comparison"""
    fields = {}

    # Extract title
    title_list = item.get('http://purl.org/dc/elements/1.1/title', [])
    if title_list and isinstance(title_list, list) and len(title_list) > 0:
        fields['title'] = title_list[0].get('@value', '')

    # Extract publisher/publication place
    publisher_list = item.get('http://purl.org/dc/elements/1.1/publisher', [])
    if publisher_list and isinstance(publisher_list, list) and len(publisher_list) > 0:
        fields['publisher'] = publisher_list[0].get('@value', '')

    # Extract date
    date_fields = ['http://purl.org/dc/elements/1.1/date',
                   'http://purl.org/dc/elements/1.1/non_standard_date',
                   'http://purl.org/dc/elements/1.1/start_date',
                   'http://purl.org/dc/elements/1.1/publication_date']

    for date_field in date_fields:
        date_list = item.get(date_field, [])
        if date_list and isinstance(date_list, list) and len(date_list) > 0:
            fields['date'] = date_list[0].get('@value', '')
            break

    # Extract author/creator
    creator_list = item.get('http://purl.org/dc/elements/1.1/contributor', [])
    if not creator_list:
        creator_list = item.get('http://purl.org/dc/elements/1.1/creator', [])
    if creator_list and isinstance(creator_list, list) and len(creator_list) > 0:
        fields['author'] = creator_list[0].get('@value', '')

    return fields


def filter_results_by_criteria(api_results, csv_row, search_type):
    """סינון תוצאות לפי קריטריונים: תאריך בלבד"""
    if not api_results:
        return []

    filtered_results = []

    for api_result in api_results:
        api_fields = extract_api_fields(api_result)

        # בדיקת תאריך - חייב להיות לפני 1942
        if not is_valid_date(api_fields.get('date', '')):
            print(f"  מסונן בגלל תאריך: {api_fields.get('date', '')}")
            continue

        # הסרת בדיקת מקום פרסום - כל מקום מותר עכשיו

        # אם עבר את כל הבדיקות - להוסיף לתוצאות
        result_with_match_info = {
            'api_title': api_fields.get('title', ''),
            'api_record_id': '',
            'api_id': api_result.get('@id', ''),
            'api_publisher': api_fields.get('publisher', ''),
            'api_date': api_fields.get('date', ''),
            'api_author': api_fields.get('author', ''),
        }

        # Extract record ID properly
        record_id_list = api_result.get('http://purl.org/dc/elements/1.1/recordid', [])
        if record_id_list and isinstance(record_id_list, list) and len(record_id_list) > 0:
            result_with_match_info['api_record_id'] = record_id_list[0].get('@value', '')

        filtered_results.append(result_with_match_info)

    return filtered_results


def extract_result_data(result: Dict, csv_row: pd.Series, search_type: str) -> List[Dict]:
    """Extract title and record_id from API result - limit to top 5 results and filter"""
    extracted_results = []

    if not result:
        print("No result to extract from")
        return extracted_results

    # Handle different response formats
    docs = []
    if isinstance(result, dict) and 'docs' in result:
        docs = result['docs']
        print(f"Found 'docs' key with {len(docs)} items")
    elif isinstance(result, list):
        docs = result
        print(f"Result is a list with {len(docs)} items")
    else:
        print(f"Unexpected result format: {type(result)}")
        return extracted_results

    print(f"Processing all {len(docs)} results before filtering")

    # Filter results based on date and publishing place criteria
    filtered_docs = filter_results_by_criteria(docs, csv_row, search_type)

    print(f"After filtering: {len(filtered_docs)} results match criteria")

    # Limit to top 30 after filtering
    filtered_docs = filtered_docs[:30]
    print(f"Taking top {len(filtered_docs)} filtered results")

    for i, result_data in enumerate(filtered_docs):
        if i < 3:
            print(
                f"Item {i}: title='{result_data['api_title'][:50]}...', record_id='{result_data['api_record_id']}', publisher='{result_data['api_publisher']}', date='{result_data['api_date']}'")

        extracted_results.append(result_data)

    print(f"Final extracted results: {len(extracted_results)}")
    return extracted_results

def generate_search_combinations(row: pd.Series) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Generate single search combination: Author + Book Title only"""
    combinations = []

    # עיבוד וניקוי הנתונים
    book_title = clean_book_title(row.get('Book Title'))  # שימוש בפונקציה החדשה
    author = expand_abbreviations(clean_search_term(row.get('Author')))  # הרחבת קיצורים

    # אם אין שם ספר - לא לבצע שום חיפוש
    if not book_title:
        print("  Book Title is empty - skipping all searches for this row")
        return combinations

    # רק קומבינציה אחת: Author + Book Title
    if book_title and author:
        combinations.append(("Author + Book Title",
                             [("creator", author), ("title", book_title)]))
    elif book_title:
        # אם אין מחבר, רק שם הספר
        combinations.append(("Book Title Only",
                             [("title", book_title)]))

    return combinations


def process_enhanced_search(csv_file_path: str, start_row: int = 0, stop_row: int = None):
    """Process the CSV file with enhanced search capabilities"""
    df = pd.read_csv(csv_file_path)

    if stop_row is None:
        stop_row = len(df)

    start_row = max(0, start_row)
    stop_row = min(len(df), stop_row)

    if start_row >= stop_row:
        print("Invalid row range. Start row must be less than stop row.")
        return

    print(f"Processing rows {start_row} to {stop_row - 1} (total: {stop_row - start_row} rows)")

    df_slice = df.iloc[start_row:stop_row].copy()
    all_results = []

    # יצירת progress bar
    with tqdm(total=len(df_slice), desc="Processing rows", unit="rows") as pbar:
        for idx, (index, row) in enumerate(df_slice.iterrows()):
            # עדכון progress bar עם מידע נוכחי
            pbar.set_description(f"Processing row {index + 1}/{len(df)} (Index: {index})")

            print(f"\nProcessing row {index + 1}/{len(df)} (Index: {index})")

            # בדיקה ראשונית - אם אין Book Title, לדלג על השורה
            book_title = clean_book_title(row.get('Book Title'))
            if not book_title:
                print(f"  Skipping row {index} - Book Title is empty")
                row_dict = row.to_dict()
                row_dict.update({
                    'search_type': 'Skipped - No Book Title',
                    'api_title': '',
                    'api_record_id': '',
                    'api_id': '',
                    'api_publisher': '',
                    'api_date': '',
                    'api_author': ''
                })
                all_results.append(row_dict)
                pbar.update(1)
                continue

            # בדיקה אם יש כבר תוצאות בעמודה api_title
            existing_api_title = row.get('api_title')
            if existing_api_title and str(existing_api_title).strip() != '' and not pd.isna(existing_api_title):
                print(f"  Skipping row {index} - Already has API results")
                row_dict = row.to_dict()
                all_results.append(row_dict)
                pbar.update(1)
                continue

            search_combinations = generate_search_combinations(row)

            if not search_combinations:
                print(f"No valid search combinations for row {index}")
                row_dict = row.to_dict()
                row_dict.update({
                    'search_type': 'No valid search',
                    'api_title': '',
                    'api_record_id': '',
                    'api_id': '',
                    'api_publisher': '',
                    'api_date': '',
                    'api_author': ''
                })
                all_results.append(row_dict)
                pbar.update(1)
                continue

            row_has_results = False
            for search_desc, search_params in search_combinations:
                print(f"  Trying: {search_desc}")

                # שימוש בחיפוש פשוט במקום מורכב
                search_term = search_params[0][1]  # לקחת את המחרוזת הפשוטה
                result = search_nli_simple(search_term)

                if result is not None:
                    extracted_results = extract_result_data(result, row, search_desc)

                    if extracted_results:
                        row_has_results = True
                        print(f"    Found {len(extracted_results)} matching results after filtering")

                        for api_result in extracted_results:
                            row_dict = row.to_dict()
                            row_dict['search_type'] = search_desc
                            row_dict.update(api_result)
                            all_results.append(row_dict)
                    else:
                        print(f"    No results passed filtering criteria")
                else:
                    print(f"    No API response")

                time.sleep(1)

            if not row_has_results:
                row_dict = row.to_dict()
                row_dict.update({
                    'search_type': 'No results found',
                    'api_title': '',
                    'api_record_id': '',
                    'api_id': '',
                    'api_publisher': '',
                    'api_date': '',
                    'api_author': ''
                })
                all_results.append(row_dict)

            # עדכון progress bar
            pbar.update(1)
            remaining = len(df_slice) - (idx + 1)
            pbar.set_postfix({
                'Completed': idx + 1,
                'Remaining': remaining,
                'Results': len(all_results)
            })

    results_df = pd.DataFrame(all_results)
    output_file = csv_file_path.replace('.csv', f'_enhanced_search_results_{start_row}_{stop_row}.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nEnhanced search results saved to: {output_file}")
    print(f"Total result rows: {len(results_df)}")

    return results_df


def test_api_connection():
    """Test API connection with a simple known query"""
    print("Testing API connection...")
    current_api_key = get_current_api_key()

    test_query = "jerusalem"
    url = f"{BASE_URL}?api_key={current_api_key}&query=title,contains,{test_query}&index=0&sort=rank"

    try:
        print(f"Test URL: {url}")
        response = requests.get(url, timeout=30)
        print(f"Test response status: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print("API connection successful!")
                return True
            except json.JSONDecodeError:
                print("API responded but with invalid JSON")
                return False
        else:
            print(f"API test failed with status {response.status_code}")
            return False

    except Exception as e:
        print(f"API test exception: {e}")
        return False


def main():
    """Main function to run the enhanced search"""
    if not test_api_connection():
        print("API connection test failed. Please check your API keys and try again.")
        return

    csv_file_path = input("Enter the path to your CSV file: ").strip()
    if not csv_file_path:
        csv_file_path = '/Users/noamkashi/Downloads/likutay_clean.csv'

    try:
        start_row = int(input("Enter start row (0-based index, default 0): ") or "0")
        stop_row_input = input("Enter stop row (leave empty for all rows): ").strip()
        stop_row = int(stop_row_input) if stop_row_input else None
    except ValueError:
        print("Invalid input. Using default values.")
        start_row = 0
        stop_row = None

    print(f"Starting with API key: {get_current_api_key()[:5]}...")

    try:
        results = process_enhanced_search(csv_file_path, start_row, stop_row)
        print("Process completed successfully!")
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()