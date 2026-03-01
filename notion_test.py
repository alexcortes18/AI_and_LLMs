import argparse
import os

from dotenv import load_dotenv
from notion_client import Client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Notion API connectivity test.")
    parser.add_argument("--page-id", help="Notion page ID to retrieve.")
    parser.add_argument("--database-id", help="Notion database ID to query.")
    return parser


def main() -> None:
    load_dotenv()
    notion_api_key = os.getenv("NOTION_API_KEY")
    if not notion_api_key:
        raise SystemExit("Missing NOTION_API_KEY in environment or .env file.")

    parser = build_parser()
    args = parser.parse_args()
    if not args.page_id and not args.database_id:
        raise SystemExit("Provide --page-id or --database-id.")

    notion = Client(auth=notion_api_key)

    if args.page_id:
        page = notion.pages.retrieve(page_id=args.page_id)
        print(f"Page access OK. Page id: {page['id']}")

    if args.database_id:
        result = notion.databases.query(database_id=args.database_id)
        print(f"Database access OK. Rows returned: {len(result['results'])}")


if __name__ == "__main__":
    main()
